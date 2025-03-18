#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:04:12 2025

@author: andrey
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from UNet import UNet
from train_and_evaluate import train, evaluate
from utilities import prepare_pet_dataset


# Create directories for checkpoints and results
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)

 

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable benchmark mode in cuDNN for performance improvement
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Hyperparameters
    num_epochs = 10
    learning_rate = 0.001
    batch_size = 8
    num_classes = 3  # Oxford-IIIT Pet Dataset has 3 classes: background, pet, and pet edge
    image_size = 128  # Fixed image size for resizing

    # Dataset preparation
    data_dir = './data'
    train_dataset, val_dataset = prepare_pet_dataset(data_dir, image_size)
    
    # Inspect target values
    #inspect_target_values(train_dataset)
    
    # Reduce the number of workers to avoid excessive worker creation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Model, criterion, and optimizer
    model = UNet(in_channels=3, out_channels=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # For mixed precision training

    # Training and evaluation
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, scaler, epoch)
        evaluate(model, device, val_loader, criterion)

        # Save model checkpoint
        torch.save(model.state_dict(), f'checkpoints/unet_epoch_{epoch}.pth')

        # Visualize predictions
        visualize_predictions(model, device, val_loader, epoch)

if __name__ == "__main__":
    main()