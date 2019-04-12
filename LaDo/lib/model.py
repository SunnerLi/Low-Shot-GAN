from lib.module import ResBlockDiscriminator, FirstResBlockDiscriminator, ResBlockOrigin
from lib.utils  import weights_init_Xavier
from lib.layer  import SpectralNorm2d
import torch.nn.functional as F
import torch.nn as nn
import torch

"""
    This script defines the main model which will be used in LaDo, including:
        1. (Appearance or Content) Encoder
        2. Seperate Generator
        3. Discriminator

    @Author: Shuian-Kai Kao
    @Revise: Cheng-Che Lee
"""

class Encoder(nn.Module):
    def __init__(self, z_dims, channels=3):
        """
            The constructor of (Appearance or Content) Encoder

            Arg:    z_dims      (Int)   - The size of latent representation
                    channels    (Int)   - The number of channel in the image
        """
        super().__init__()
        self.conv0 = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(channels, 128, 3, stride=1, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(128, 128, 3, stride=1, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(128, 128, 3, stride=1, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
        )
        self.conv3 = nn.Sequential(
            SpectralNorm2d(nn.Conv2d(128, 128, 3, stride=1, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
            SpectralNorm2d(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)).apply(weights_init_Xavier)),
            nn.LeakyReLU(0.2),
        )
        self.final_mean = nn.Sequential(
            nn.Conv2d(128, z_dims, kernel_size=4, stride=1, padding=0),
        )
        self.final_var = nn.Sequential(
            nn.Conv2d(128, z_dims, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, img):
        out = self.conv0(img)           # 128 x 32 x 32
        out = self.conv1(out)           # 128 x 16 x 16
        out = self.conv2(out)           # 128 x 8 x 8
        out = self.conv3(out)           # 128 x 4 x 4
        mean = self.final_mean(out)     # 128 x 1 x 1
        logvar = self.final_var(out)    # 128 x 1 x 1
        return mean, logvar


class Generator(nn.Module):
    def __init__(self, z_dims, sn_norm=False):
        """
            The constructor of Generator

            Arg:    z_dims      (Int)   - The size of latent representation
                    sn_norm     (Bool)  - If adopting spectral normalization or not
        """
        super().__init__()
        if sn_norm:
            self.block0 = nn.Sequential(
                SpectralNorm2d(nn.Linear(z_dims, 128 * 4 * 4)),
                nn.BatchNorm1d(128 * 4 * 4),
                nn.ReLU()
            )
        else:
            self.block0 = nn.Sequential(
                nn.Linear(z_dims, 128 * 4 * 4),
                nn.BatchNorm1d(128 * 4 * 4),
                nn.ReLU()
            )
      
        self.block1 = ResBlockOrigin(128, 128, stride=2, sn_norm=sn_norm)
        self.block2 = ResBlockOrigin(128, 128, stride=2, sn_norm=sn_norm)
        self.block3 = ResBlockOrigin(128, 128, stride=2, sn_norm=sn_norm)
        self.block4 = ResBlockOrigin(128, 128, stride=2, sn_norm=sn_norm)
        if sn_norm:
            self.block5 = nn.Sequential(
                SpectralNorm2d(nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)),
                nn.Tanh()
            )
        else:
            self.block5 = nn.Sequential(
                nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            )

    def forward(self, z):
        f0 = self.block0(z.view(len(z),-1))
        f1 = self.block1(f0.view(len(z),-1,4,4))
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        img = self.block5(f4)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        """
            The constructor of Discriminator
        """
        super().__init__()
        self.model = nn.Sequential(
            FirstResBlockDiscriminator(3, 128, stride=2),     # 128 x 32 x 32
            ResBlockDiscriminator(128, 128, stride=2),        # 128 x 16 x 16
            ResBlockDiscriminator(128, 128, stride=2),        # 128 x 8 x 8
            ResBlockDiscriminator(128, 128, stride=2),        # 128 x 4 x 4
            ResBlockDiscriminator(128, 128, stride=1),        # 128 x 4 x 4
            ResBlockDiscriminator(128, 128, stride=1),        # 128 x 4 x 4
            nn.ReLU(),
            nn.AvgPool2d(4)
        )
        self.fc = SpectralNorm2d(nn.Linear(128, 1))

    def forward(self, img):
        output = self.fc(self.model(img).view(-1, 128))
        return output.squeeze(1).squeeze()