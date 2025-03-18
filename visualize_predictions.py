#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:14:57 2025

@author: andrey
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

# Visualization function
def visualize_predictions(model, device, val_loader, epoch):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()
            data = data.cpu().numpy()
            target = target.cpu().numpy()
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
                plt.savefig(f'results/epoch_{epoch}_sample_{idx * len(output) + i}.png')
                plt.close()