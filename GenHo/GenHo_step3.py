#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 08:12:18 2018

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
from collections import OrderedDict

"""
    *********************************************************************************************
    * The main code for the paper : Learning Few-Shot Generative Networks for Cross-Domain Data *
    *********************************************************************************************
    According to the paper, the GenHo approach contains 5 steps. 
    This script will train the 3rd step for GenHo. 
    @Author: Hsuan-Kai Kao
"""

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']='0'


parser = argparse.ArgumentParser()

# Hyper-parameter
parser.add_argument('--img_size'                    , type = int, default = 64)
parser.add_argument('--z_dims'                      , type = int, default = 128)
parser.add_argument('--batch_size'                  , type = int, default = 64)
parser.add_argument('--total_epoch'                 , type = int, default = 100)
parser.add_argument('--content_layer'               , type = int, default = 8)

# Load Path
parser.add_argument('--tar_path'              , type = str, required = True)
parser.add_argument('--step1_path'            , type = str, default = 'Save_model/Step1/step1_100.pth')
parser.add_argument('--step2_path'            , type = str, default = 'Save_model/Step2/step2_100.pth')

# Save Path
parser.add_argument('--image_path'            , type = str, default = 'Img/Step3')
parser.add_argument('--model_path'            , type = str, default = 'Save_model/Step3')
args = parser.parse_args()

if os.path.exists(args.image_path) is False:
    os.makedirs(args.image_path)
if os.path.exists(args.model_path) is False:
    os.makedirs(args.model_path)
    
# target data prepare
tar_img = Mydataset(args.tar_path, args.img_size)
train_data = DataLoader(tar_img, batch_size=args.batch_size, shuffle=True)

# Test data
for i,t in enumerate(train_data):
    if i ==0:  
        test_data = t
        save_image(test_data[:25], args.image_path + '/real.png', nrow=5, normalize=True)
        test_data = test_data.cuda()

# load step1 & step2 weight
load_weight1 = torch.load(args.step1_path)
load_weight2 = torch.load(args.step2_path)

# Model initialization
Ec = Encoder(layer=args.content_layer, sn_norm=False).cuda()
D2 = DiscriminatorRes().cuda()
Ga2 = GeneratorResA(layer=args.content_layer, sn_norm=False).cuda()
    
D2.apply(weights_init('xavier'))
Ga2.apply(weights_init('xavier'))

Ec.load_state_dict(load_weight2['Ec'])
for p in Ec.parameters():
    p.requires_grad = False

# load Ga2 weight
state = Ga2.state_dict()
Ga_parameters = [(k,v) for k,v in load_weight1['G1'].items() if k in state]
Ga_parameters = OrderedDict(Ga_parameters)
state.update(Ga_parameters)
Ga2.load_state_dict(state)

optimizer_D2 = optim.Adam(filter(lambda p: p.requires_grad, D2.parameters()), lr=0.0002, betas=(0.0,0.9))
optimizer_Ga2 = optim.Adam(filter(lambda p: p.requires_grad, Ga2.parameters()), lr=0.0002, betas=(0.0,0.9))

scheduler_d2 = optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.99)
scheduler_ga2 = optim.lr_scheduler.ExponentialLR(optimizer_Ga2, gamma=0.99)

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
mse_loss = torch.nn.MSELoss().cuda()
vgg_loss = VGGLoss().cuda()

# Training
total = math.ceil(len(tar_img) / args.batch_size)
G_losses = []
D_losses = []
Re_losses = []
Gen_img = []
for epoch in range(args.total_epoch):
    print('\nEpoch%d' %(epoch+1))
    G_loss = []
    D_loss = []
    Re_loss = []
    for i,x2 in enumerate(train_data):
        
        # Real data
        x2 = x2.cuda()
        mini_batch = x2.shape[0]

        ####################################################################### 
        # Train Discriminator for target Data
        #######################################################################
        for _ in range(1):
            D2.zero_grad()
            
            # Adversarial ground truths
            y_real = torch.ones(mini_batch).cuda()
            y_fake = torch.zeros(mini_batch).cuda()
            
            # Adversarial loss
            fake_fs = Ec(x2)
            fake_imgs = Ga2(fake_fs)
            real_logit = D2(x2).squeeze()
            fake_logit = D2(fake_imgs.detach()).squeeze()
            D_real_loss = adversarial_loss(real_logit, y_real)
            D_fake_loss = adversarial_loss(fake_logit, y_fake)
            d_loss = (D_real_loss + D_fake_loss) / 2
            
            # WGAN-GP
#            wd = real_logit.mean() - fake_logit.mean()
#            gp = calc_gradient_penalty(D1, x1, fake_imgs, mini_batch, 10.)
#            d_loss = -wd + gp
            
            # Update discriminator
            d_loss.backward()
            optimizer_D2.step()
            
        #######################################################################
        # Train Generator for target data
        #######################################################################
        for _ in range(1):
            Ec.zero_grad()
            Ga2.zero_grad()
            
            # Adversarial loss
            fake_fs = Ec(x2)
            fake_imgs = Ga2(fake_fs)
            fake_logit = D2(fake_imgs).squeeze()
            g_loss = adversarial_loss(fake_logit, y_real)
            
            # WGAN-GP
#            g_loss = -fake_logit.mean()
            
            # Reconstruction loss
            re_loss = vgg_loss(fake_imgs,x2)
#           re_loss = MSE_loss(fake_imgs,x2)
            
            # Update generator
            eg_loss =  0.5 * re_loss + 0.5 * g_loss
            eg_loss.backward()
            optimizer_Ga2.step()

        # Iteration loss
        D_loss.append(d_loss.item())
        G_loss.append(g_loss.item())
        Re_loss.append(re_loss.item())
        
        # Process bar
        percent = math.ceil(((i+1) / total) * 100)
        bar = '#' * int(percent/4) + ' ' * (25-int(percent/4))
        sys.stdout.write('\r' + 'training step ' + bar + '[%d%%]' %percent)
        sys.stdout.flush()
        
        if i%10==0: 
            sys.stdout.write(' Re_loss: %f D_loss %f G_loss: %f' %(np.mean(Re_loss), np.mean(D_loss), np.mean(G_loss)))
            sys.stdout.flush()
        
    scheduler_d2.step()
    scheduler_ga2.step()
    
    # Epoch loss
    D_losses.append(np.mean(D_loss))
    G_losses.append(np.mean(G_loss))
    Re_losses.append(np.mean(Re_loss))
    
    gen_imgs  = Ga2(Ec(test_data))
    if (epoch+1) % 5 ==0:    
        Gen_img.append(gen_imgs.data[5])
    save_image(gen_imgs.data[:25], args.image_path + '/step3_%d.png' % (epoch+1), nrow=5,  normalize=True)
    

    # do checkpointing
    print('\nSave model...\n')
    states = {
            'epoch': epoch + 1,
            'D2': D2.state_dict(),
            'Ga2': Ga2.state_dict(),
            'optimizer_D2': optimizer_D2.state_dict(),
            'optimizer_Ga2': optimizer_Ga2.state_dict(),
            'scheduler_d2': scheduler_d2.state_dict(),
            'scheduler_ga2': scheduler_ga2.state_dict(),
            'Gen_img': Gen_img,
            }
    torch.save(states, args.model_path + '/step3_%d.pth' %(epoch+1))
    
Gen_img = torch.stack(Gen_img)
save_image(Gen_img.data, args.image_path + '/show.png', nrow=5, normalize=True)

plt.figure()
plt.plot(D_losses)
plt.title('D loss')
plt.savefig(args.image_path + '/d_loss.png')

plt.figure()
plt.plot(G_losses)
plt.title('G loss')
plt.savefig(args.image_path + '/g_loss.png')

plt.figure()
plt.plot(Re_losses)
plt.title('Re loss')
plt.savefig(args.image_path + '/re_loss.png')
        
        
        
        









