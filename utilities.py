#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:11:59 2025

@author: andrey
"""

import torch
from torchvision import datasets, transforms
import numpy as np


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
    
    train_dataset = datasets.OxfordIIITPet(root=data_dir, split='trainval', download=True, target_types=['segmentation'], transform=transform, target_transform=target_transform)
    val_dataset = datasets.OxfordIIITPet(root=data_dir, split='test', download=True, target_types=['segmentation'], transform=transform, target_transform=target_transform)
    
    return train_dataset, val_dataset
