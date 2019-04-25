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
from torch.utils.data import DataLoader, sampler
from torchvision.utils import save_image
from model import *
from utils import *
from collections import OrderedDict

"""
    *********************************************************************************************
    * The main code for the paper : Learning Few-Shot Generative Networks for Cross-Domain Data *
    *********************************************************************************************
    According to the paper, the GenHo approach contains 5 steps. 
    This script will train the 2nd step for GenHo. 
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
parser.add_argument('--pair_iter'                   , type = int, default = 50)
parser.add_argument('--content_layer'               , type = int, default = 8)

# Load Path
parser.add_argument('--src_path'              , type = str, required = True)
parser.add_argument('--src_pair'              , type = str, required = True)
parser.add_argument('--tar_pair'              , type = str, required = True)
parser.add_argument('--step1_path'            , type = str, default = 'Save_model/Step1/step1_100.pth')

# Save Path
parser.add_argument('--image_path'            , type = str, default = 'Img/Step2')
parser.add_argument('--model_path'            , type = str, default = 'Save_model/Step2')
args = parser.parse_args()

if os.path.exists(args.image_path) is False:
    os.makedirs(args.image_path)
if os.path.exists(args.model_path) is False:
    os.makedirs(args.model_path)

# Source data
src_img = Mydataset(args.src_path, args.img_size)

# pair data prepare
pair_src = Mydataset(args.src_pair, args.img_size)
pair_tar = Mydataset(args.tar_pair, args.img_size)

indices = list(range(len(src_img)))
np.random.seed(35)
np.random.shuffle(indices)
train_indices = indices[64:]
test_indices = indices[0:64]
train_sampler = sampler.SubsetRandomSampler(train_indices)
test_sampler = sampler.SubsetRandomSampler(test_indices)
train_data = DataLoader(src_img, batch_size=args.batch_size, sampler=train_sampler)
test_data = DataLoader(src_img, batch_size=args.batch_size, sampler=test_sampler)
pair_source = DataLoader(pair_src, batch_size=args.batch_size, shuffle=False)
pair_target = DataLoader(pair_tar, batch_size=args.batch_size, shuffle=False)

# Test data prepare    
for x in test_data:
    test = x

# Pair data prepare  
p1 = []  
p2 = []
for s,t in zip(pair_source,pair_target):
    p1.append(s.data)
    p2.append(t.data)
p1 = torch.cat(p1)
p2 = torch.cat(p2)
save_image(p1[:25], 'src.png', nrow=5, normalize=True)
save_image(p2[:25], 'tar.png', nrow=5, normalize=True)

# load source generator weight
load_weight = torch.load(args.step1_path)

# Model Initialization
Ec = Encoder(layer=args.content_layer, sn_norm=False).cuda()
Dc = DiscriminatorResC(layer=args.content_layer, sn_norm=True).cuda()
Gc = GeneratorResC(args.z_dims, layer=args.content_layer, sn_norm=False).cuda()
Ga = GeneratorResA(layer=args.content_layer, sn_norm=False).cuda()

Ec.apply(weights_init('xavier'))
Dc.apply(weights_init('xavier'))

optimizer_Ec = optim.Adam(filter(lambda p: p.requires_grad, Ec.parameters()), lr=0.0002, betas=(0.0,0.9))
optimizer_Dc = optim.Adam(filter(lambda p: p.requires_grad, Dc.parameters()), lr=0.0002, betas=(0.0,0.9))

scheduler_ec = optim.lr_scheduler.ExponentialLR(optimizer_Ec, gamma=0.99)
scheduler_dc = optim.lr_scheduler.ExponentialLR(optimizer_Dc, gamma=0.99)

# load Gc weight
state = Gc.state_dict()
Gc_parameters = [(k,v) for k,v in load_weight['G1'].items() if k in state]
Gc_parameters = OrderedDict(Gc_parameters)
state.update(Gc_parameters)
Gc.load_state_dict(state)

for p in Gc.parameters():
    p.requires_grad = False

# load Ga weight
state = Ga.state_dict()
Ga_parameters = [(k,v) for k,v in load_weight['G1'].items() if k in state]
Ga_parameters = OrderedDict(Ga_parameters)
state.update(Ga_parameters)
Ga.load_state_dict(state)

for p in Ga.parameters():
    p.requires_grad = False

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
mse_loss = torch.nn.MSELoss().cuda()
vgg_loss = VGGLoss().cuda()
l1_loss = torch.nn.L1Loss().cuda()
#Contrasitive_loss = ContrastiveLoss()

# Training
total = math.ceil(len(src_img) / args.batch_size)
Re_losses = []
Ec_losses = []
Dc_losses = []
Pair_loss = []
Gen_img = []
print(Ec)   
save_image(test[:25], args.image_path + '/real.png', nrow=5, normalize=True)

for epoch in range(args.total_epoch):
    print('Epoch%d' %(epoch+1))
    Re_loss = []
    Ec_loss = []
    Dc_loss = []
    for i, train in enumerate(train_data):
            
        # Real data
        x1 = train
        x1 = x1.cuda()
        mini_batch = x1.shape[0]
        
        # Adversarial ground truths
        y_real = torch.ones(mini_batch).cuda()
        y_fake = torch.zeros(mini_batch).cuda()
        
        #######################################################################
        # Train Content Discriminator
        #######################################################################
        for _ in range(1):
            Gc.zero_grad()
            Dc.zero_grad()
            
            z = torch.randn(mini_batch, args.z_dims, 1, 1).cuda()
            real_fs = Gc(z)
            fake_fs = Ec(x1)
            real_logit = Dc(real_fs).squeeze()
            fake_logit = Dc(fake_fs.detach()).squeeze()
            
            # WGAN-GP
#            wd = real_logit.mean() - fake_logit.mean()
#            gp = calc_gradient_penalty(Ds, real_fs, fake_fs, mini_batch, 10.)
#            d_loss = -wd + gp
#            d_loss.backward()
            
            # Adversarial loss
            d_real = adversarial_loss(real_logit, y_real)
            d_fake = adversarial_loss(fake_logit, y_fake)
            d_loss = (d_real + d_fake) / 2
            d_loss.backward()
            
            optimizer_Dc.step()
            
        #######################################################################0
        # Train Content Encoder
        #######################################################################
        for _ in range(1): 
            Ec.zero_grad()
            Ga.zero_grad()
            
            # Content loss
            fake_fs = Ec(x1)
            fake_logit = Dc(fake_fs).squeeze()
            ec_loss = adversarial_loss(fake_logit, y_real)
            
            # WGAN-GP
#            es_loss = -fake_logit.mean()
            
            # Reconstruction loss
            fake_fs = Ec(x1)
            fake_imgs = Ga(fake_fs)
            re_loss = vgg_loss(fake_imgs, x1)
#            e_loss = mse_loss(fake_imgs,x1)
#            e_loss = l1_loss(fake_imgs,x1)
            
            # Update Encoder
            e_loss = 0.25 * ec_loss + 0.75 * re_loss
            e_loss.backward()
        
        
        # Pair loss
        if i % args.pair_iter ==0:
            idx = torch.randperm(len(p1))
            pair1 = p1[idx]
            pair2 = p2[idx]
            source_fs = Ec(pair1[:args.batch_size].cuda())
            target_fs = Ec(pair2[:args.batch_size].cuda())
            p_loss = 0.0001 * mse_loss(source_fs, target_fs)
            p_loss.backward()
            
        optimizer_Ec.step()   
        
        # Iteration loss   
        Re_loss.append(re_loss.item())
        Ec_loss.append(ec_loss.item())
        Dc_loss.append(d_loss.item())
        
        # Process bar
        percent = math.ceil(((i+1) / total) * 100)
        bar = '#' * int(percent/4) + ' ' * (25-int(percent/4))
        sys.stdout.write('\r' + 'training step ' + bar + '[%d%%]' %percent)
        sys.stdout.flush()
        
        if i%10==0: 
            sys.stdout.write(' Re_loss: %f Ec_loss %f Dc_loss: %f Pair_loss: %f' %(np.mean(Re_loss), np.mean(Ec_loss), np.mean(Dc_loss),
                            p_loss.item()))
            sys.stdout.flush()
        
    scheduler_ec.step()    
    scheduler_dc.step()
    
    # Epoch loss
    Dc_losses.append(np.mean(Dc_loss))
    Ec_losses.append(np.mean(Ec_loss))
    Re_losses.append(np.mean(Re_loss))
    Pair_loss.append(p_loss.item())
    
    gen_imgs  = Ga(Ec(test.cuda()))
    if (epoch+1) % 5 ==0:    
        Gen_img.append(gen_imgs.data[5])
    save_image(gen_imgs.data[:25], args.image_path + '/step2_%d.png' % (epoch+1), nrow=5, normalize=True)
    
    # do checkpointing
    print('\nSave model...\n')
    states = {
            'epoch': epoch + 1,
            'Ec': Ec.state_dict(),
            'Dc': Dc.state_dict(),
            'optimizer_Ec': optimizer_Ec.state_dict(),
            'optimizer_Dc': optimizer_Dc.state_dict(),
            'scheduler_ec': scheduler_ec.state_dict(),
            'scheduler_dc': scheduler_dc.state_dict(),
            'Gen_img': Gen_img,
            'Re_losses': Re_losses,
            'Ec_losses': Ec_losses,
            'Dc_losses': Dc_losses,
            'Pair_losses': Pair_loss,
            'test': test
            }
    torch.save(states, args.model_path + '/step2_%d.pth' %(epoch+1))
    
Gen_img = torch.stack(Gen_img)
save_image(Gen_img.data, args.image_path + '/show.png', nrow=5, normalize=True)

plt.figure()
plt.plot(Re_losses)
plt.title('Re loss')
plt.savefig(args.image_path + '/re_loss.png')

plt.figure()
plt.plot(Ec_losses)
plt.title('Ec loss')
plt.savefig(args.image_path + '/ec_loss.png')

plt.figure()
plt.plot(Dc_losses)
plt.title('Dc loss')
plt.savefig(args.image_path + '/d_loss.png')

plt.figure()
plt.plot(Pair_loss)
plt.title('Pair loss')
plt.savefig(args.image_path + '/pair_loss.png')