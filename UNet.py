#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:04:18 2025

@author: andrey
"""


import torch.nn as nn


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
