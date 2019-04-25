#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:06:44 2018

@author: Shiuan
"""

from __future__ import print_function
import os
import sys
import math
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import *
from utils import *

"""
    *********************************************************************************************
    * The main code for the paper : Learning Few-Shot Generative Networks for Cross-Domain Data *
    *********************************************************************************************
    According to the paper, the GenHo approach contains 5 steps. 
    This script will train the 1st step for GenHo. 
    @Author: Hsaun-Kai Kao
"""


warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser()

# Hyper-parameter
parser.add_argument('--img_size'                    , type = int, default = 64)
parser.add_argument('--z_dims'                      , type = int, default = 128)
parser.add_argument('--batch_size'                  , type = int, default = 64)
parser.add_argument('--total_epoch'                 , type = int, default = 100)

# Load Path
parser.add_argument('--src_path'              , type = str, required = True)

# Save Path
parser.add_argument('--image_path'            , type = str, default = 'Img/Step1')
parser.add_argument('--model_path'            , type = str, default = 'Save_model/Step1')
args = parser.parse_args()

# Create folder
if os.path.exists(args.image_path ) is False:
    os.makedirs(args.image_path)
if os.path.exists(args.model_path) is False:
    os.makedirs(args.model_path)

# Source data
src_img = Mydataset(args.src_path, args.img_size)
train_data = DataLoader(src_img, batch_size=args.batch_size, shuffle=True)

# Model initialization
D1 = DiscriminatorRes().cuda()
G1 = GeneratorRes(args.z_dims, sn_norm=False).cuda()
D1.apply(weights_init('xavier'))
G1.apply(weights_init('xavier'))

optimizer_D1 = optim.Adam(filter(lambda p: p.requires_grad, D1.parameters()), lr=0.0002, betas=(0.0,0.9))
optimizer_G1 = optim.Adam(filter(lambda p: p.requires_grad, G1.parameters()), lr=0.0002, betas=(0.0,0.9))

scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_D1, gamma=0.99)
scheduler_g1 = optim.lr_scheduler.ExponentialLR(optimizer_G1, gamma=0.99)

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Training
total = math.ceil(len(src_img) / args.batch_size)
fixed_z = torch.randn(args.batch_size, args.z_dims, 1, 1).cuda()
D_losses = []
G_losses = []
Gen_img = []
for epoch in range(args.total_epoch):
    print('Epoch%d' %(epoch+1))
    G_loss = []
    D_loss = []
    for i,x1 in enumerate(train_data):
        
        # Process bar
        percent = int(((i+1) / total) * 100)
        bar = '#' * int(percent/2) + ' ' * (50-int(percent/2))
        sys.stdout.write('\r' + 'training step ' + bar + '[%d%%]' %percent)
        sys.stdout.flush()
        
        if i%10==0: 
            sys.stdout.write('  D_loss: %f G_loss: %f' %(np.mean(D_loss), np.mean(G_loss)))
            sys.stdout.flush()
            
        # Real data
        x1 = x1.cuda()
        mini_batch = x1.shape[0]
        
        ####################################################################### 
        # Train Discriminator
        #######################################################################
        for _ in range(1):
            D1.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(mini_batch, args.z_dims, 1, 1).cuda()
            
            # Adversarial ground truths
            y_real = torch.ones(mini_batch).cuda()
            y_fake = torch.zeros(mini_batch).cuda()
            
            # Adversarial loss
            fake_imgs = G1(z)[5]
            real_logit = D1(x1).squeeze()
            fake_logit = D1(fake_imgs.detach()).squeeze()
            D_real_loss = adversarial_loss(real_logit, y_real)
            D_fake_loss = adversarial_loss(fake_logit, y_fake)
            d_loss = (D_real_loss + D_fake_loss) / 2
            
            # hinge loss
#            D_real_loss = torch.nn.ReLU()(1.0 - real_logit).mean()
#            D_fake_loss = torch.nn.ReLU()(1.0 + fake_logit).mean()
#            d_loss = (D_real_loss + D_fake_loss)
            
            # WGAN-GP
#            wd = real_logit.mean() - fake_logit.mean()
#            gp = calc_gradient_penalty(D1, x1, fake_imgs, mini_batch, 10.)
#            d_loss = -wd + gp
            
            # Update discriminator
            d_loss.backward()
            optimizer_D1.step()
            
        #######################################################################
        # Train Generator
        #######################################################################
        for _ in range(1):
            G1.zero_grad()
            
            # Sample noise as generator input
            z = torch.randn(mini_batch, args.z_dims, 1, 1).cuda()
            
            # Gan loss
            fake_imgs = G1(z)[5]
            fake_logit = D1(fake_imgs).squeeze()
            g_loss = adversarial_loss(fake_logit, y_real)
            
            # hinge loss & WGAN-GP
#            g_loss = -fake_logit.mean()
            
            # Update generator
            g_loss.backward()
            optimizer_G1.step()
        
        # Iteration loss
        D_loss.append(d_loss.item())
        G_loss.append(g_loss.item())
        
        
    scheduler_d.step()
    scheduler_g1.step()
          
    # Epoch loss
    D_losses.append(np.mean(D_loss))
    G_losses.append(np.mean(G_loss))
    
    print("\nD loss: %f g loss: %f" %(np.mean(D_loss), np.mean(G_loss)))

    gen_imgs = G1(fixed_z)[5]
    if (epoch+1) % 4 ==0:    
        Gen_img.append(gen_imgs.data[5])
    save_image(gen_imgs.data[:25], args.image_path + '/step1_%d.png' % (epoch+1), nrow=5, normalize=True)
    
    # do checkpointing
    print('Save model...\n')
    states = {
            'epoch': epoch + 1,
            'D1': D1.state_dict(),
            'G1': G1.state_dict(),
            'optimizer_D1': optimizer_D1.state_dict(),
            'optimizer_G1': optimizer_G1.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_g1': scheduler_g1.state_dict(),
            'fixed_z': fixed_z,
            'D_losses': D_losses,
            'G_losses': G_losses,
            'Gen_img': Gen_img
            }
    torch.save(states, args.model_path + '/step1_%d.pth' %(epoch+1))

Gen_img = torch.stack(Gen_img)
save_image(Gen_img.data, args.image_path + '/show.png', nrow=5, normalize=True)

plt.figure()
plt.plot(D_losses)
plt.title('D loss:')
plt.savefig(args.image_path + '/d_loss.png')

plt.figure()
plt.plot(G_losses)
plt.title('G loss')
plt.savefig(args.image_path + '/g_loss.png')