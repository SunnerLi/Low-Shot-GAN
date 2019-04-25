#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:06:44 2018

@author: Shiuan
"""

from __future__ import print_function
import os
import math
import random
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
    This script will train the 5th step for GenHo. 
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
parser.add_argument('--content_layer'               , type = int, default = 8)

# Load Path
parser.add_argument('--step1_path'            , type = str, default = 'Save_model/Step1/step1_100.pth')
parser.add_argument('--step3_path'            , type = str, default = 'Save_model/Step3/step3_100.pth')
parser.add_argument('--prior_path'            , type = str, default = 'Save_model/Step4/10000z.npz')

# Save Path
parser.add_argument('--image_path'            , type = str, default = 'Img/Step5')
parser.add_argument('--model_path'            , type = str, default = 'Save_model/Step5')
args = parser.parse_args()


if os.path.exists(args.image_path) is False:
    os.makedirs(args.image_path)
if os.path.exists(args.model_path) is False:
    os.makedirs(args.model_path)

# Data prepare
data = np.load(args.prior_path)
target_data, prior = data['target_data'], data['prior']


# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()
vgg_loss = VGGLoss().cuda()
MSE_loss = torch.nn.L1Loss().cuda()

# Model initialization
D2 = DiscriminatorRes().cuda()
Gc = GeneratorResC(args.z_dims, layer=args.content_layer, sn_norm=False).cuda()
Ga2 = GeneratorResA(layer=args.content_layer, sn_norm=False).cuda()

# Initialize weights
D2.apply(weights_init('xavier'))
Gc.apply(weights_init('xavier'))
Ga2.apply(weights_init('xavier'))

# load source generator weight
load_weight = torch.load(args.step1_path)
Ga_parameters = torch.load(args.step3_path)['Ga2']
D2_parameters = torch.load(args.step3_path)['D2']

# load Gc weight
state = Gc.state_dict()
Gc_parameters = [(k,v) for k,v in load_weight['G1'].items() if k in state]
Gc_parameters = OrderedDict(Gc_parameters)
state.update(Gc_parameters)
Gc.load_state_dict(state)
for p in Gc.parameters():
    p.requires_grad = False

# load Ga2 weight
Ga2.load_state_dict(Ga_parameters)

# load D2 weight         
D2.load_state_dict(load_weight['D1'])

optimizer_D2 = optim.Adam(filter(lambda p: p.requires_grad, D2.parameters()), lr=0.0002, betas=(0.0,0.9))
optimizer_Gc = optim.Adam(Gc.parameters(), lr=0.0002, betas=(0.0, 0.9))
optimizer_Ga2 = optim.Adam(Ga2.parameters(), lr=0.0002, betas=(0.0, 0.9))

scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.99)
scheduler_gc = optim.lr_scheduler.ExponentialLR(optimizer_Gc, gamma=0.99)
scheduler_ga2 = optim.lr_scheduler.ExponentialLR(optimizer_Ga2, gamma=0.99)

# Training
total = math.ceil(len(target_data) / args.batch_size)
random_z = torch.randn(args.batch_size, args.z_dims, 1, 1).cuda()
ground_truth = torch.Tensor(target_data[:25])
save_image(ground_truth, args.image_path + '/ground_truth.png', nrow=5, normalize=True)
D_losses = []
G_losses = []
Re_losses = []
Gen_img = []
for epoch in range(args.total_epoch):
    print('Epoch%d' %(epoch+1))
    G_loss = []
    D_loss = []
    Re_loss = []
    idx = np.arange(len(target_data))
    random.shuffle(idx)
    target_data = target_data[idx]
    prior = prior[idx]
    
    total = math.ceil(len(target_data) / args.batch_size)
    for i in range(total):            
        # real data
        x2 = target_data[i*args.batch_size:(i+1)*args.batch_size]
        x2 = torch.tensor(x2).cuda()
        z = prior[i*args.batch_size:(i+1)*args.batch_size]
        z = torch.tensor(z).cuda()
        mini_batch = x2.shape[0]
        ####################################################################### 
        # train D1
        #######################################################################
        for _ in range(1):
            D2.zero_grad()
            
            # Adversarial ground truths
            y_real = torch.ones(mini_batch).cuda()
            y_fake = torch.zeros(mini_batch).cuda()
            
            # Adversarial loss
            fake_fs = Gc(z)
            fake_imgs = Ga2(fake_fs)
            real_logit = D2(x2).squeeze()
            fake_logit = D2(fake_imgs.detach()).squeeze()
            D_real_loss = adversarial_loss(real_logit, y_real)
            D_fake_loss = adversarial_loss(fake_logit, y_fake)
            d_loss = (D_real_loss + D_fake_loss) / 2
            
            # WGAN-GP
#            wd = real_logit.mean() - fake_logit.mean()
#            gp = calc_gradient_penalty(D2, x2, fake_imgs, mini_batch, 10.)
#            d_loss = -wd + gp
            
            # Update discriminator
            d_loss.backward()
            optimizer_D2.step()
            
        #######################################################################
        # train G1
        #######################################################################
        for _ in range(1):       
            Gc.zero_grad()
            Ga2.zero_grad()
            
            # Adversarial loss
            fake_fs = Gc(z)
            fake_imgs = Ga2(fake_fs)
            fake_logit = D2(fake_imgs).squeeze()
            g_loss = adversarial_loss(fake_logit, y_real)
#            g_loss = -fake_logit.mean()
            
            # reconstruction loss
            re_loss = vgg_loss(fake_imgs, x2)
#            re_loss = MSE_loss(fake_imgs, x1)
            
            rg_loss = 0.5 * re_loss + 0.5 * g_loss
            rg_loss.backward()
            optimizer_Gc.step()
            optimizer_Ga2.step()
        
        # Iteration loss
        D_loss.append(d_loss.item())
        G_loss.append(g_loss.item())
        Re_loss.append(re_loss.item())
        
    scheduler_d.step()
    scheduler_gc.step()
    scheduler_ga2.step()
    
    # Epoch loss
    D_losses.append(np.mean(D_loss))
    G_losses.append(np.mean(G_loss))
    Re_losses.append(np.mean(Re_loss))
    print("D loss: %f G loss: %f Re loss: %f" %(np.mean(D_loss), np.mean(G_loss), np.mean(Re_loss)))
    
    gen_imgs = Ga2(Gc(random_z))
    if (epoch+1) % 4 ==0:    
        Gen_img.append(gen_imgs.data[5])
    save_image(gen_imgs.data[:25], args.image_path + '/step5_%d.png' % (epoch+1), nrow=5, normalize=True)
    
    # do checkpointing
    print('Save model...\n')
    states = {
            'epoch': epoch + 1,
            'D2': D2.state_dict(),
            'Gc': Gc.state_dict(),
            'Ga2': Ga2.state_dict(),
            'optimizer_D2': optimizer_D2.state_dict(),
            'optimizer_Gc': optimizer_Gc.state_dict(),
            'optimizer_Ga2': optimizer_Ga2.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_gc': scheduler_gc.state_dict(),
            'scheduler_ga2': scheduler_ga2.state_dict(),
            'D_losses': D_losses,
            'G_losses': G_losses,
            'Re_losses': Re_losses,
            'Gen_img': Gen_img
            }
    torch.save(states, args.model_path + '/step5_%d.pth' %(epoch+1))


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

plt.figure()
plt.plot(Re_losses)
plt.title('Re loss')
plt.savefig(args.image_path + '/re_loss.png')