#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:06:44 2018

@author: Shiuan
"""

from __future__ import print_function
import os
import warnings
import math
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import *
from utils import *
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors

"""
    *********************************************************************************************
    * The main code for the paper : Learning Few-Shot Generative Networks for Cross-Domain Data *
    *********************************************************************************************
    According to the paper, the GenHo approach contains 5 steps. 
    This script will train the 4th step for GenHo. 
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
parser.add_argument('--sample'                      , type = int, default = 10000)

# Load Path
parser.add_argument('--src_pair'              , type = str, required = True)
parser.add_argument('--tar_pair'              , type = str, required = True)
parser.add_argument('--step1_path'            , type = str, default = 'Save_model/Step1/step1_100.pth')
parser.add_argument('--step2_path'            , type = str, default = 'Save_model/Step2/step2_100.pth')

# Save Path
parser.add_argument('--image_path'            , type = str, default = 'Img/Step4')
parser.add_argument('--prior_path'            , type = str, default = 'Save_model/Step4')
args = parser.parse_args()

if os.path.exists(args.image_path) is False:
    os.makedirs(args.image_path)
if os.path.exists(args.prior_path) is False:
    os.makedirs(args.prior_path)

# pair data prepare
pair_src = Mydataset(args.src_pair, args.img_size)
pair_tar = Mydataset(args.tar_pair, args.img_size)
        
source_data = DataLoader(pair_src, batch_size=args.batch_size, shuffle=False)
target_data = DataLoader(pair_tar, batch_size=args.batch_size, shuffle=False) 

ground_truth_s = []
for d in source_data:
    ground_truth_s.append(d)
ground_truth_s = torch.cat(ground_truth_s)

ground_truth_t = []
for d in target_data:
    ground_truth_t.append(d)
ground_truth_t = torch.cat(ground_truth_t)

save_image(ground_truth_s[:25], args.image_path + '/ground_truth_s.png', nrow=5, normalize=True)
save_image(ground_truth_t[:25], args.image_path + '/ground_truth_t.png', nrow=5, normalize=True)
    
# load step1 & step2 weight
load_weight1 = torch.load(args.step1_path)
load_weight2 = torch.load(args.step2_path)

# Model Initialization
Ec = Encoder(layer=args.content_layer, sn_norm=False)
Gc = GeneratorResC(args.z_dims, layer=args.content_layer, sn_norm=False)
Ga = GeneratorResA(layer=args.content_layer, sn_norm=False)

# load Ec weight
Ec.load_state_dict(load_weight2['Ec'])

# load Gc weight
state = Gc.state_dict()
Gc_parameters = [(k,v) for k,v in load_weight1['G1'].items() if k in state]
Gc_parameters = OrderedDict(Gc_parameters)
state.update(Gc_parameters)
Gc.load_state_dict(state)

for p in Gc.parameters():
    p.requires_grad = False

# load Ga weight
state = Ga.state_dict()
Ga_parameters = [(k,v) for k,v in load_weight1['G1'].items() if k in state]
Ga_parameters = OrderedDict(Ga_parameters)
state.update(Ga_parameters)
Ga.load_state_dict(state)

for p in Ga.parameters():
    p.requires_grad = False

Ec.eval()
Gc.eval()
Ga.eval() 
    
# sample z
fixed_z = torch.randn(args.sample, args.z_dims, 1, 1)
total = math.ceil(len(fixed_z) / args.batch_size)

# sample content distribution
sample = []
for i in range(total):
    sample.append(Gc(fixed_z[i*args.batch_size:(i+1)*args.batch_size]))
sample = torch.cat(sample)

# source content distribution
s = []
for d in source_data:
    s.append(Ec(d))
s = torch.cat(s)

# target content distribution
t = []
for d in target_data:
    t.append(Ec(d))
t = torch.cat(t)


sample = sample.data.cpu().numpy()
s = s.data.cpu().numpy()
t = t.data.cpu().numpy()
sample = sample.reshape(sample.shape[0],-1)
s = s.reshape(s.shape[0],-1)
t = t.reshape(t.shape[0],-1)

nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='euclidean').fit(sample)

# nearest neighbors for source
distances, indices = nbrs.kneighbors(s)
prior_s = fixed_z[indices[:,0]]
prior_img_s = Ga(Gc(prior_s))

# nearest neighbors for target
distances, indices = nbrs.kneighbors(t)
prior_t = fixed_z[indices[:,0]]
prior_img_t = Ga(Gc(prior_t))


np.savez(args.prior_path  + '/' + str(args.sample) + 'z.npz', target_data=ground_truth_t, prior=prior_t)
save_image(prior_img_s.data[:25], args.image_path + '/prior_img_s'  + '_' + str(args.sample) + 'z.png', nrow=5, normalize=True)
save_image(prior_img_t.data[:25], args.image_path + '/prior_img_t'  + '_' + str(args.sample) + 'z.png', nrow=5, normalize=True)
