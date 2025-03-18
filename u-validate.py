#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:05:06 2025

@author: andrey
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# U-Net model definition
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            self.double_conv(in_channels, 64),
            self.down(64, 128),
            self.down(128, 256),
            self.down(256, 512),
            self.down(512, 1024),
        )
        self.decoder = nn.Sequential(
            self.up(1024, 512),
            self.up(512, 256),
            self.up(256, 128),
            self.up(128, 64),
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
            self.double_conv(in_channels // 2, out_channels)
        )

    def forward(self, x):
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        bottleneck = self.encoder[4](enc4)
        dec1 = self.decoder[0](bottleneck)
        dec2 = self.decoder[1](dec1 + enc4)
        dec3 = self.decoder[2](dec2 + enc3)
        dec4 = self.decoder[3](dec3 + enc2)
        return self.final_conv(dec4 + enc1)

# Custom transform to handle both image and mask
class ToTensorTarget:
    def __call__(self, image, target):
        return transforms.ToTensor()(image), torch.tensor(np.array(target), dtype=torch.long)

# Dataset preparation
def prepare_pet_dataset(data_dir, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Lambda(lambda img: torch.tensor(np.array(img) - 1, dtype=torch.long))  # Adjust target values
    ])
    
    val_dataset = datasets.OxfordIIITPet(root=data_dir, split='test', download=True, target_types=['segmentation'], transform=transform, target_transform=target_transform)
    return val_dataset

# Visualization function
def visualize_predictions(model, device, val_loader):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()
            data = data.cpu().numpy()
            target = target.cpu().numpy()
            if idx>2: break
            for i in range(len(output)):
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 3, 1)
                plt.imshow(np.transpose(data[i], (1, 2, 0)))
                plt.title('Input Image')
                plt.subplot(1, 3, 2)
                plt.imshow(target[i], cmap='gray')
                plt.title('Ground Truth')
                plt.subplot(1, 3, 3)
                plt.imshow(output[i], cmap='gray')
                plt.title('Prediction')
                plt.savefig(f'results/validation_sample_{idx * len(output) + i}.png')
                plt.close()

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 8
    num_classes = 3  # Oxford-IIIT Pet Dataset has 3 classes: background, pet, and pet edge
    image_size = 128  # Fixed image size for resizing

    # Dataset preparation
    data_dir = './data'
    val_dataset = prepare_pet_dataset(data_dir, image_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)

    # Model
    model = UNet(in_channels=3, out_channels=num_classes).to(device)

    # Load checkpoint
    checkpoint_path = 'checkpoints/unet_epoch_7.pth'  # Replace with your checkpoint path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Visualize predictions
    visualize_predictions(model, device, val_loader)

if __name__ == "__main__":
    main()