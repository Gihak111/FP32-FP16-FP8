import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import psutil
import os
from tabulate import tabulate
import logging

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt'),
        logging.StreamHandler()
    ]
)

# FP8 Emulation with Calibration (E4M3 format)
def quantize_to_fp8(tensor, use_calibration=False):
    """
    Quantizes a tensor to FP8 (E4M3 format) with optional calibration.
    Calibration computes a dynamic scale based on tensor min-max range.
    """
    try:
        if use_calibration:
            # Calibration: Compute scale based on min and max of the tensor
            min_val = tensor.min()
            max_val = tensor.max()
            abs_max = max(abs(min_val), abs(max_val))
            scale = 240.0 / abs_max if abs_max != 0 else 1.0  # 240 is max E4M3 value
        else:
            scale = 1.0  # No calibration

        tensor_scaled = tensor * scale
        max_val = 240.0  # Max representable value in E4M3
        min_normal = 0.00390625  # Smallest normal number in E4M3
        
        tensor_scaled = torch.clamp(tensor_scaled, -max_val, max_val)
        
        sign = torch.sign(tensor_scaled)
        abs_tensor = torch.abs(tensor_scaled)
        exponent = torch.floor(torch.log2(abs_tensor + 1e-10))  # Avoid log2(0)
        mantissa = abs_tensor / (2 ** exponent)
        mantissa = torch.round(mantissa * 8) / 8  # 3-bit mantissa
        quantized = sign * mantissa * (2 ** exponent)
        
        quantized[abs_tensor < min_normal] = 0.0
        return quantized / scale
    except Exception as e:
        logging.error(f"Error in FP8 quantization: {e}")
        raise

# Simple Neural Network for MNIST
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x, use_fp8=False, use_calibration=False):
        try:
            x = x.view(-1, 784)
            if use_fp8:
                x = quantize_to_fp8(x, use_calibration)
                self.fc1.weight.data = quantize_to_fp8(self.fc1.weight.data, use_calibration)
            x = self.fc1(x)
            x = self.relu(x)
            if use_fp8:
                x = quantize_to_fp8(x, use_calibration)
                self.fc2.weight.data = quantize_to_fp8(self.fc2.weight.data, use_calibration)
            x = self.fc2(x)
            x = self.relu(x)
            if use_fp8:
                x = quantize_to_fp8(x, use_calibration)
                self.fc3.weight.data = quantize_to_fp8(self.fc3.weight.data, use_calibration)
            x = self.fc3(x)
            return x
        except Exception as e:
            logging.error(f"Error in forward pass: {e}")
            raise

# Training function
def train_model(model, train_loader, criterion, optimizer, device, use_fp8=False, use_fp16=False, use_calibration=False, epochs=5):
    model.train()
    start_time = time.time()
    total_memory = 0
    scaler = GradScaler() if use_fp16 and device.type == 'cuda' else None
    
    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = len(train_loader)
        for i, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_fp16 and device.type == 'cuda':
                with autocast():
                    outputs = model(images, use_fp8=use_fp8, use_calibration=use_calibration)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if use_fp8:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = quantize_to_fp8(param.grad.data, use_calibration)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, use_fp8=use_fp8, use_calibration=use_calibration)
                loss = criterion(outputs, labels)
                loss.backward()
                if use_fp8:
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data = quantize_to_fp8(param.grad.data, use_calibration)
                optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                logging.info(f'Epoch {epoch+1}, Batch {i}/{batch_count}, Loss: {loss.item():.4f}')
                print(f'Epoch {epoch+1}/{epochs}, Batch {i}/{batch_count}, Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / batch_count
        precision = "FP16" if use_fp16 else ("FP8" if use_fp8 else "FP32")
        logging.info(f'Epoch {epoch+1}, {precision} Average Loss: {avg_loss:.4f}')
        print(f'Epoch {epoch+1}/{epochs}, {precision} Average Loss: {avg_loss:.4f}')
        
        memory = sum(p.element_size() * p.numel() for p in model.parameters()) / 1024
        if use_fp8:
            memory *= 0.25
        elif use_fp16:
            memory *= 0.5
        total_memory = max(total_memory, memory)
    
    training_time = time.time() - start_time
    logging.info(f'{precision} Total Training Time: {training_time:.2f} seconds')
    print(f'{precision} Total Training Time: {training_time:.2f} seconds')
    return total_memory, training_time

# Evaluation function
def evaluate_model(model, test_loader, device, use_fp8=False, use_fp16=False, use_calibration=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if use_fp16 and device.type == 'cuda':
                with autocast():
                    outputs = model(images, use_fp8=use_fp8, use_calibration=use_calibration)
            else:
                outputs = model(images, use_fp8=use_fp8, use_calibration=use_calibration)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    precision = "FP16" if use_fp16 else ("FP8" if use_fp8 else "FP32")
    logging.info(f'{precision} Test Accuracy: {accuracy:.2f}%')
    print(f'{precision} Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')
    if device.type != 'cuda':
        logging.warning('FP16 training is less effective on CPU. GPU recommended.')
        print('Warning: FP16 training is less effective on CPU. GPU recommended.')
    
    # Load MNIST dataset
    logging.info('Loading MNIST dataset...')
    print('Loading MNIST dataset...')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    results = []
    
    # FP32 Training
    logging.info('Starting FP32 Training')
    print('Starting FP32 Training...')
    model_fp32 = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_fp32 = optim.Adam(model_fp32.parameters(), lr=0.001)
    memory_fp32, time_fp32 = train_model(model_fp32, train_loader, criterion, optimizer_fp32, device, use_fp8=False, use_fp16=False, use_calibration=False)
    accuracy_fp32 = evaluate_model(model_fp32, test_loader, device, use_fp8=False, use_fp16=False, use_calibration=False)
    results.append(['FP32', accuracy_fp32, memory_fp32, time_fp32])
    
    # FP16 Training
    logging.info('Starting FP16 Training')
    print('Starting FP16 Training...')
    model_fp16 = SimpleNN().to(device)
    optimizer_fp16 = optim.Adam(model_fp16.parameters(), lr=0.001)
    memory_fp16, time_fp16 = train_model(model_fp16, train_loader, criterion, optimizer_fp16, device, use_fp8=False, use_fp16=True, use_calibration=False)
    accuracy_fp16 = evaluate_model(model_fp16, test_loader, device, use_fp8=False, use_fp16=True, use_calibration=False)
    results.append(['FP16', accuracy_fp16, memory_fp16, time_fp16])
    
    # FP8 Training with Calibration
    logging.info('Starting FP8 Training with Calibration')
    print('Starting FP8 Training with Calibration...')
    model_fp8 = SimpleNN().to(device)
    optimizer_fp8 = optim.Adam(model_fp8.parameters(), lr=0.001)
    memory_fp8, time_fp8 = train_model(model_fp8, train_loader, criterion, optimizer_fp8, device, use_fp8=True, use_fp16=False, use_calibration=True)
    accuracy_fp8 = evaluate_model(model_fp8, test_loader, device, use_fp8=True, use_fp16=False, use_calibration=True)
    results.append(['FP8 (Calibrated)', accuracy_fp8, memory_fp8, time_fp8])
    
    # Display results
    headers = ['Precision', 'Accuracy (%)', 'Memory (KB)', 'Training Time (s)']
    table = tabulate(results, headers=headers, tablefmt='grid', floatfmt='.2f')
    logging.info('\nResults Comparison:\n' + table)
    print('\nResults Comparison:')
    print(table)

if __name__ == '__main__':
    main()