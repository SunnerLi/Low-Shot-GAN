#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 21:04:20 2018

@author: Shiuan
"""

from __future__ import print_function
import torch.nn as nn
from spectral_normalization import SpectralNorm
from blocks import *

def Linear(in_features, out_features, sn_norm):
    if sn_norm:
        linear = SpectralNorm(nn.Linear(in_features, out_features))
    else:
        linear = nn.Linear(in_features, out_features)

    return linear

def Conv2d(in_channels, out_channels, kernel=3, stride=1, padding=1, sn_norm=False):
    if sn_norm:
        conv = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel, stride, padding))
    else:
        conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
    
    return conv
   
###############################################################################
# ResNetwork
############################################################################### 
# Whole Generator
class GeneratorRes(nn.Module):
    def __init__(self, z_dims, sn_norm=False):
        super(GeneratorRes, self).__init__()
        
        # 128 * 4 * 4
        self.block0 = nn.Sequential(
                Linear(z_dims, 512 * 4 * 4, sn_norm=sn_norm),
                nn.BatchNorm1d(512 * 4 * 4),
                nn.ReLU()
                )
        
        # 128 * 8 * 8
        self.block1 = ResBlockOrigin(512, 512, stride=2, sn_norm=sn_norm)
        
        # 128 * 16 * 16
        self.block2 = ResBlockOrigin(512, 256, stride=2, sn_norm=sn_norm)
        
        # 128 * 32 * 32
        self.block3 = ResBlockOrigin(256, 128, stride=2, sn_norm=sn_norm)
        
        # 128 * 64 * 64
        self.block4 = ResBlockOrigin(128, 64, stride=2, sn_norm=sn_norm)
        
        # 3 * 64 * 64
        self.block5 = nn.Sequential(
                Conv2d(64, 3, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                nn.Tanh()
                )

    def forward(self, z):
        f0 = self.block0(z.view(len(z),-1))
        f1 = self.block1(f0.view(len(z),-1,4,4))
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        img = self.block5(f4)
        return f0, f1, f2, f3, f4, img    

# Content Generator
class GeneratorResC(nn.Module):
    def __init__(self, z_dims, layer=8, sn_norm=False):
        super(GeneratorResC, self).__init__()
        
        self.layer = layer
        
        if layer>=8:
            # 512 * 4 * 4
            self.block0 = nn.Sequential(
                Linear(z_dims, 512 * 4 * 4, sn_norm=sn_norm),
                nn.BatchNorm1d(512 * 4 * 4),
                nn.ReLU()
                )
        
            # 512 * 8 * 8
            self.block1 = ResBlockOrigin(512, 512, stride=2, sn_norm=sn_norm)
        
        if layer>=16:
            # 256 * 16 * 16
            self.block2 = ResBlockOrigin(512, 256, stride=2, sn_norm=sn_norm)
        
        if layer>=32:
            # 128 * 32 * 32
            self.block3 = ResBlockOrigin(256, 128, stride=2, sn_norm=sn_norm)

    def forward(self, z):
        if self.layer>=8:   
            f = self.block0(z.view(len(z),-1))
            f = self.block1(f.view(len(z),-1,4,4))
        if self.layer>=16:
            f = self.block2(f)
        if self.layer>=32:
            f = self.block3(f)
        return f

# Appearance Generator
class GeneratorResA(nn.Module):
    def __init__(self, layer=8, sn_norm=False):
        super(GeneratorResA, self).__init__()
        
        self.layer = layer
        
        if layer<16:
            # 256 * 16 * 16
            self.block2 = ResBlockOrigin(512, 256, stride=2, sn_norm=sn_norm)
        
        if layer<32:
            # 128 * 32 * 32
            self.block3 = ResBlockOrigin(256, 128, stride=2, sn_norm=sn_norm)
        
        if layer<64:
            # 64 * 64 * 64
            self.block4 = ResBlockOrigin(128, 64, stride=2, sn_norm=sn_norm)
            
            # 3 * 64 * 64
            self.block5 = nn.Sequential(
                        Conv2d(64, 3, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                        nn.Tanh()
                        )       
        
    def forward(self, f):
        if self.layer<16:    
            f = self.block2(f)
        if self.layer<32:    
            f = self.block3(f)
        if self.layer<64:
            f = self.block4(f)
        img = self.block5(f)
        return img

# Discriminator for image
class DiscriminatorRes(nn.Module):
    def __init__(self):
        super(DiscriminatorRes, self).__init__()
        
        self.model = nn.Sequential(
                
                # 128 x 32 x 32
                FirstResBlockDiscriminator(3, 64, stride=2),
                
                # 128 x 16 x 16
                ResBlockDiscriminator(64, 128, stride=2),
                
                # 128 x 8 x 8
                ResBlockDiscriminator(128, 256, stride=2),
                
                # 128 x 4 x 4
                ResBlockDiscriminator(256, 512, stride=2),
                
                # 128 x 4 x 4
                ResBlockDiscriminator(512, 512, stride=1),
                ResBlockDiscriminator(512, 512, stride=1),
                nn.ReLU(),
                nn.AvgPool2d(4)
                )
        
        self.fc = SpectralNorm(nn.Linear(512, 1))
#        self.conv = SpectralNorm(nn.Conv2d(128, 1, 1, 1, 0))

    def forward(self, img):
        output = self.fc(self.model(img).view(-1, 512))
#        output = self.conv(self.model(img)).view(-1, 1)
        return output.squeeze(1)

# Discriminator for Content Distribution
class DiscriminatorResC(nn.Module):
    def __init__(self, layer=8, sn_norm=True):
        super(DiscriminatorResC, self).__init__()

        self.model = []
        
        if layer>=32:
            # 128 x 16 x 16
            self.model.append(ResBlockEncoder(128, 256, stride=2, sn_norm=sn_norm))
            
        if layer>=16:
            # 128 x 8 x 8
            self.model.append(ResBlockEncoder(256, 512, stride=2, sn_norm=sn_norm))
            
        if layer>=8:
            # 128 x 4 x 4
            self.model.append(ResBlockEncoder(512, 512, stride=2, sn_norm=sn_norm))
            self.model.append(ResBlockEncoder(512, 512, stride=1, sn_norm=sn_norm))
#            self.model.append(ResBlockEncoder(128, 128, stride=1, sn_norm=sn_norm))
            self.model.append(nn.AvgPool2d(4))
       
        self.model = nn.Sequential(*self.model)
        self.fc = Linear(512, 1, sn_norm=sn_norm)
#        self.conv = Conv2d(128, 1, sn_norm=sn_norm)
            
    def forward(self, f):
        output = self.fc(self.model(f).view(-1, 512))
#        output = self.conv(self.model(f))
        return output.squeeze(1)

# Content Encoder with Resblock
class EncoderRes(nn.Module):
    def __init__(self, layer=8, sn_norm=False):
        super(EncoderRes, self).__init__()
        
        self.layer = layer
        
        if layer==32:
            # 128 x 32 x 32
            self.block0 = FirstResBlockEncoder(3, 128, stride=2, sn_norm=sn_norm)
        
        if layer==16:
            # 128 x 16 x 16
            self.block0 = FirstResBlockEncoder(3, 128, stride=2, sn_norm=sn_norm)
            self.block1 = ResBlockEncoder(128, 256, stride=2, sn_norm=sn_norm)
        
        if layer==8:
            # 128 x 8 x 8
            self.block0 = FirstResBlockEncoder(3, 128, stride=2, sn_norm=sn_norm)
            self.block1 = ResBlockEncoder(128, 256, stride=2, sn_norm=sn_norm)
            self.block2 = ResBlockEncoder(256, 512, stride=2, sn_norm=sn_norm)

    def forward(self, x):
        if self.layer<64:    
            f = self.block0(x)
        if self.layer<32:    
            f = self.block1(f)
        if self.layer<16:    
            f = self.block2(f)
        return f

# Content Encoder    
class Encoder(nn.Module):
    def __init__(self, layer=8, sn_norm=False):
        super(Encoder, self).__init__()
        
        self.layer = layer
        
        if layer==32:
            # 128 * 32 * 32             
            self.block0 = nn.Sequential(
                    Conv2d(3, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=4, stride=2, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
        
        if layer==16:            
            # 128 * 16 * 16
            self.block0 = nn.Sequential(
                    Conv2d(3, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=4, stride=2, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
            self.block1 = nn.Sequential(
                    Conv2d(128, 256, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(256, 256, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(256, 256, kernel=4, stride=2, padding=1, sn_norm=sn_norm),
                    #nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
        
        if layer==8:
            # 128 * 8 * 8
            self.block0 = nn.Sequential(
                    Conv2d(3, 128, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(128, 128, kernel=4, stride=2, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
            self.block1 = nn.Sequential(
                    Conv2d(128, 256, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(256, 256, kernel=4, stride=2, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
            self.block2 = nn.Sequential(
                    Conv2d(256, 512, kernel=3, stride=1, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    Conv2d(512, 512, kernel=4, stride=2, padding=1, sn_norm=sn_norm),
#                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    )
            
    def forward(self, x):
        if self.layer<64:
            f = self.block0(x)
        if self.layer<32:
            f = self.block1(f)
        if self.layer<16:
            f = self.block2(f)     
        return f