from lib.layer import SpectralNorm2d
import torch.nn.functional as F 
import torch.nn as nn
import numpy as np
import torch

"""
    This script defines modules which will be used in the model of LaDo

    @Author: Shuian-Kai Kao
    @Revise: Cheng-Che Lee
"""

class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Define main pass
        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
                nn.ReLU(),
                SpectralNorm2d(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1))
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
                nn.ReLU(),
                SpectralNorm2d(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )

        # Define by pass 
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class FirstResBlockDiscriminator(nn.Module):
    """
        This class defines the special ResBlock just for the first layer of the discriminator
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
            nn.ReLU(),
            SpectralNorm2d(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class ResBlockOrigin(nn.Module):
    """ 
        This nested class define the residual block which will be used in generator
    """
    def __init__(self, in_channels, out_channels, stride = 1, sn_norm=True):
        super().__init__()
        
        if sn_norm:
            if stride == 1:
                self.model = nn.Sequential(
                    SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    SpectralNorm2d(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
                    nn.BatchNorm2d(out_channels),
                )
            
                self.bypass = nn.Sequential()
            if stride != 1:
                self.model = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    SpectralNorm2d(nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)),
                    nn.BatchNorm2d(out_channels),
                )
                
                self.bypass = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0))
                        )
        else:
            if stride == 1:
                self.model = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                    nn.BatchNorm2d(out_channels),
                )
            
                self.bypass = nn.Sequential()
            if stride != 1:
                self.model = nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                    nn.BatchNorm2d(out_channels),
                )
                
                self.bypass = nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
                        )

    def forward(self, x):
        return F.relu(self.model(x) + self.bypass(x))