#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:09:51 2025

@author: andrey
"""

import torch
from tqdm import tqdm  # Import tqdm for progress bar



# Training function
def train(model, device, train_loader, optimizer, criterion, scaler, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):  # Add tqdm progress bar
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(data)
            loss = criterion(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# Evaluation function
def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Evaluating"):  # Add tqdm progress bar
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
    val_loss /= len(val_loader)
    print(f'\nValidation set: Average loss: {val_loss:.4f}\n')
    return val_loss