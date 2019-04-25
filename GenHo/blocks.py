#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:06:44 2018

@author: Shiuan
"""

from torch import nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm

def Conv2d(in_channels, out_channels, kernel=3, stride=1, padding=1, sn_norm=False):
    if sn_norm:
        conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
    
    return conv

class ResBlockOrigin(nn.Module):
    """ 
        Deep Residual Learning for Image Recognition https://arxiv.org/abs/1512.03385
    """
    def __init__(self, in_channels, out_channels, stride = 1, sn_norm=False):
        super(ResBlockOrigin, self).__init__()
        
        self.model = []
        self.bypass = []
        if stride != 1:
            self.model.append(nn.Upsample(scale_factor=2))
            self.bypass.append(nn.Upsample(scale_factor=2))
        
        self.model += [
            Conv2d(in_channels, out_channels, 3, 1, padding=1, sn_norm=sn_norm),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            Conv2d(out_channels, out_channels, 3, 1, padding=1, sn_norm=sn_norm),
            nn.BatchNorm2d(out_channels)
            ]
        
#        if stride != 1 or in_channels != out_channels:
        self.bypass.append(Conv2d(in_channels, out_channels, 1, 1, padding=0, sn_norm=sn_norm))
        
        self.model = nn.Sequential(*self.model)
        self.bypass = nn.Sequential(*self.bypass)

    def forward(self, x):
        return F.relu(self.model(x) + self.bypass(x))

# Special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
            nn.AvgPool2d(2)
            )

        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)),
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# ResBlock for the middle layer of the discriminator
class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.model = [
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1))
            ] 
        self.bypass = []
        
        if stride != 1 or in_channels != out_channels:    
            self.bypass.append(SpectralNorm(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)))
                
        if stride != 1:
            self.model.append(nn.AvgPool2d(2, stride=stride, padding=0))
            self.bypass.append(nn.AvgPool2d(2, stride=stride, padding=0))
        
        self.model = nn.Sequential(*self.model)
        self.bypass = nn.Sequential(*self.bypass)

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# Special ResBlock just for the first layer of the Encoder
class FirstResBlockEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, sn_norm=False):
        super(FirstResBlockEncoder, self).__init__()
            
        self.model = nn.Sequential(
            Conv2d(in_channels, out_channels, 3, 1, padding=1, sn_norm=sn_norm),
            nn.ReLU(),
            Conv2d(out_channels, out_channels, 3, 1, padding=1, sn_norm=sn_norm),
            nn.AvgPool2d(2)
            )
        
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            Conv2d(in_channels, out_channels, 1, 1, padding=0, sn_norm=sn_norm)
            )

    def forward(self, x):
        return F.relu(self.model(x) + self.bypass(x))

# ResBlock for the middle layer of the Encoder    
class ResBlockEncoder(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, sn_norm=False):
        super(ResBlockEncoder, self).__init__()
        
        self.model = []
        self.bypass = []     
        
        self.model = [
            Conv2d(in_channels, out_channels, 3, 1, padding=1, sn_norm=sn_norm),
            nn.ReLU(),
            Conv2d(out_channels, out_channels, 3, 1, padding=1, sn_norm=sn_norm)
            ]
        
        if stride != 1 or in_channels != out_channels:
            self.bypass.append(Conv2d(in_channels, out_channels, 1, 1, padding=0, sn_norm=sn_norm))
        
        if stride != 1:
            self.model.append(nn.AvgPool2d(2, stride=stride, padding=0))
            self.bypass.append(nn.AvgPool2d(2, stride=stride, padding=0))
        
        self.model = nn.Sequential(*self.model)
        self.bypass = nn.Sequential(*self.bypass)        
        
    def forward(self, x):
        return F.relu(self.model(x) + self.bypass(x))
        