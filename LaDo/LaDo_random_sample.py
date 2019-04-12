import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
from torchvision import transforms

from lib.utils import num2Str
from lib.lado import LaDo

from skimage import io
from tqdm import tqdm
import torch.utils.data as Data
import numpy as np
import argparse
import os

"""
    *********************************************************************************************
    * The main code for the paper : Learning Few-Shot Generative Networks for Cross-Domain Data *
    *********************************************************************************************

    This script will do the inference randomly with LaDo approach

    @Author: Cheng-Che Lee
"""

def parse():
    """
                                        Parse the argument
        =================================================================================================
            [tag]                       [type]      [default]       [function]
        =================================================================================================
            --img_size                  Int         64              The size of image
            --content_dims              Int         128             The length of content representation
            --appearance_dims           Int         64              The length of appearance representation
            --batch_size                Int         32              The batch size

            --model_path                Str         {skip}          The path of final target generator
            --root_folder               Str         {skip}          The path of output folder
            --n                         Int         64              The number of image you want to sample
        =================================================================================================
        Ret:    The argument object
    """
    parser = argparse.ArgumentParser()
    # Hyper-parameter
    parser.add_argument('--img_size'            , type = int, default = 64)
    parser.add_argument('--content_dims'        , type = int, default = 128)
    parser.add_argument('--appearance_dims'     , type = int, default = 64)
    parser.add_argument('--batch_size'          , type = int, default = 32)
    # Path
    parser.add_argument('--model_path'          , type = str, required = True)
    parser.add_argument('--root_folder'         , type = str, default  = "Training_result")
    parser.add_argument('--n'                   , type = int, default = 64)

    args = parser.parse_args()
    return args

def main(args):
    # Variable you should revise in each setting
    target_folder = os.path.join(args.root_folder, "random_sample")

    # Load model
    if not os.path.exists(target_folder):
        os.mkdir(target_folder)
    model = LaDo(args.content_dims, args.appearance_dims, args.batch_size)
    model.load(args.model_path, stage=2)

    # Sample!
    counter = 0
    for i in tqdm(range(args.n // args.batch_size + 1)):
        _, fake_tar = model.sampleN(args.batch_size)
        img = sunnertransforms.asImg(fake_tar)
        for j in range(img.shape[0]):
            io.imsave(os.path.join(target_folder, num2Str(counter) + '.png'), img[j])
            counter += 1
            if counter > args.n:
                exit()

if __name__ == '__main__':
    args = parse()
    main(args)