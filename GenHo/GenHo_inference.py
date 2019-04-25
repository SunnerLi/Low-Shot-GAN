#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 15:33:26 2019

@author: Shiuan
"""

from __future__ import print_function
import os
import math
import random
import warnings
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
    This script does the inference after step5 finished!
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
parser.add_argument('--sample_size'                 , type = int, default = 50000)

# Load Path
parser.add_argument('--step5_path'            , type = str, default = 'Save_model/Step5/step5_100.pth')

# Save Path
parser.add_argument('--inference_path'            , type = str, default = 'inference/')
args = parser.parse_args()
    

if os.path.exists(args.inference_path) is False:
    os.makedirs(args.inference_path)

# Model initialization
Gc = GeneratorResC(args.z_dims, layer=args.content_layer, sn_norm=False).cuda()
Ga2 = GeneratorResA(layer=args.content_layer, sn_norm=False).cuda()

Gc.eval()
Ga2.eval()

# load target generator weight
load_weight = torch.load(args.step5_path)

# load Gs1 weight
Gc.load_state_dict(load_weight['Gc'])

# load Ga2 weight
Ga2.load_state_dict(load_weight['Ga2'])


total_batch = math.ceil(args.sample_size/64)
c = 1
for i in range(total_batch):
    if i!=total_batch-1:
        random_z = torch.randn(args.batch_size, args.z_dims, 1, 1).cuda()
        gen_img = Ga2(Gc(random_z))
    else:
        random_z = torch.randn(args.sample_size - (args.batch_size * i), args.z_dims, 1, 1).cuda()
        gen_img = Ga2(Gc(random_z))
    for img in gen_img:
        save_image(img, args.inference_path  + '/gen_%d.png' %(c), normalize=True)
        c += 1







