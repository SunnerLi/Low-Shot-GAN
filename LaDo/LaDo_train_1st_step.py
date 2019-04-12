import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
from torchvision import transforms

from lib.utils import num2Str
from lib.lado import LaDo

from tqdm_table import tqdm_table
import torch.utils.data as Data
import numpy as np
import argparse
import torch
import os

"""
    *********************************************************************************************
    * The main code for the paper : Learning Few-Shot Generative Networks for Cross-Domain Data *
    *********************************************************************************************

    According to the paper, the LaDo approach contains 2 training step. 
    This script will train the 1st step for LaDo. 

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
            --total_epoch               Int         100             The total epoch you want to train

            --src_image_folder          Str         {skip}          The path of source image folder
            --src_pair_image_folder     Str         {skip}          The path of source pair image folder
            --tar_pair_image_folder     Str         {skip}          The path of target pair image folder
            --root_folder               Str         {skip}          The path of output folder
        =================================================================================================
        Ret:    The argument object
    """
    parser = argparse.ArgumentParser()
    # Hyper-parameter
    parser.add_argument('--img_size'                    , type = int, default = 64)
    parser.add_argument('--content_dims'                , type = int, default = 128)
    parser.add_argument('--appearance_dims'             , type = int, default = 64)
    parser.add_argument('--batch_size'                  , type = int, default = 64)
    parser.add_argument('--total_epoch'                 , type = int, default = 100)
    # Path
    parser.add_argument('--src_image_folder'            , type = str, required = True)
    parser.add_argument('--src_pair_image_folder'       , type = str, required = True)
    parser.add_argument('--tar_pair_image_folder'       , type = str, required = True)
    parser.add_argument('--root_folder'                 , type = str, default  = "Training_result")

    args = parser.parse_args()
    return args

def main(args):
    """
        Train the 1st step for LaDo

        Arg:    args    (Namespace)     - The argument object
    """
    # Create the data loader and the model
    loader = Data.DataLoader(
        dataset = sunnerData.ImageDataset(
            root = [[args.src_image_folder], [args.src_pair_image_folder], [args.tar_pair_image_folder]],
            transform = transforms.Compose([
                transforms.Resize((args.img_size, args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ]),            
            sample_method = sunnerData.OVER_SAMPLING
        ), batch_size = args.batch_size, shuffle = True, num_workers = 8
    )
    model = LaDo(args.content_dims, args.appearance_dims, args.batch_size)

    # Loop
    for ep in range(args.total_epoch + 1):
        bar = tqdm_table(loader)
        bar.set_table_setting(150)
        for i, (src_img, src_img_pair, tar_img_pair) in enumerate(bar):
            if len(src_img) == 1:
                continue
            model.setInput_1st(src_img, src_img_pair, tar_img_pair)
            if ep != 0:
                # Print total loss and update
                model.backward_1st()
                bar.set_table_info({k[14:]: v for k, v in model.getLoss(stage=1).items()})
            else:
                model.forward_1st()
                break

        # Print average loss
        if ep != 0:
            print("=====" * 20)
            print("<< Epoch {} average >>".format(ep) + "     " *20)
            string = ""
            had_write = False
            for i, (key, loss) in enumerate(model.getLoss(stage=1, normalize=True).items()):
                if loss == 0.0:
                    continue
                loss = round(loss, 10)
                if i % 2 == 1:
                    string += "{:>20}: {:>15} \t".format(key[14:], str(loss))
                    print(string)  
                    had_write = True
                else:
                    string = "{:>20}: {:>15} \t".format(key[14:], str(loss))
                    had_write = False
            if had_write == False:
                print(string) 
            model.finishEpoch()

        # Save
        if not os.path.exists(args.root_folder):
            os.mkdir(args.root_folder)
        if not os.path.exists(os.path.join(args.root_folder, 'models_1st')):
            os.mkdir(os.path.join(args.root_folder, 'models_1st'))
        model.save(path = os.path.join(args.root_folder, 'models_1st', str(ep) + '.pth'))

if __name__ == '__main__':
    args = parse()
    main(args) 