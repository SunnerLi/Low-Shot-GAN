from lib.model import Encoder, Generator, Discriminator
from lib.utils import weights_init_Xavier
from lib.loss  import KLLoss, VGGLoss

from torchvision.utils import save_image
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import itertools
import torch
import os

"""
    This script define the core part of LaDo. 
    The loss terms are list below:
        [1st step]
            1. Image Reconstruction loss
            2. Image Adversarial loss
            3. Pair Structure loss 
            4. KL loss (content and appearance)
            5. Latent Regression loss (content and appearance)

        [2nd step]
            1. Image Reconstruction loss
            2. Image Adversarial loss
            3. KL loss (appearance only)
            4. Pair Generation loss

    Loss format: loss_list_<STAGE>_<TARGET>_<LOSS_NAME>_<SUPPLEMENT>
    tensor format: <THING1>_<THING2>_<ROUND>_<SUPPLEMENT>

    @Author: Cheng-Che Lee
"""

# Define the random sampling lambda function
RANDN = lambda mean, var: torch.randn(var.size()).to(var.device) * torch.exp(var * 0.5) + mean

class LaDo(nn.Module):
    def __init__(self, content_dims, appearance_dims, batch_size):
        """
            The constructor of LaDo

            Arg:    content_dims        (Int)   - The length of content representation
                    appearance_dims     (Int)   - The length of appearance representation
                    batch_size          (Int)   - The batch size
        """
        super().__init__()
        self.content_dims = content_dims
        self.appearance_dims = appearance_dims
        self.batch_size = batch_size

        # Weight for each loss term
        self.lambda_loss_1st_ir = 0.1
        self.lambda_loss_1st_ia = 1.0
        self.lambda_loss_1st_ps = 0.001
        self.lambda_loss_1st_kl = 1e-5
        self.lambda_loss_1st_lr = 0.0001        
        self.lambda_loss_2nd_ir = 0.6
        self.lambda_loss_2nd_ia = 1.0
        self.lambda_loss_2nd_kl = 1e-5
        self.lambda_loss_2nd_pg = 0.6

        # Initialize fundamental variable
        self.loss_list_1st_ia_g = []
        self.loss_list_1st_ia_d = []
        self.loss_list_1st_ir = []
        self.loss_list_1st_ps = []
        self.loss_list_1st_kl = []
        self.loss_list_1st_lr = []
        self.loss_list_2nd_ir = []
        self.loss_list_2nd_ia_g = []
        self.loss_list_2nd_ia_d = []
        self.loss_list_2nd_kl = []
        self.loss_list_2nd_pg = []

        self.Loss_list_1st_ia_g = []
        self.Loss_list_1st_ia_d = []
        self.Loss_list_1st_ir = []
        self.Loss_list_1st_ps = []
        self.Loss_list_1st_kl = []
        self.Loss_list_1st_lr = []
        self.Loss_list_2nd_ir = []
        self.Loss_list_2nd_ia_g = []
        self.Loss_list_2nd_ia_d = []
        self.Loss_list_2nd_kl = []
        self.Loss_list_2nd_pg = []

        # Define fixed sample latent vector
        self.z_fixed = None

        # --------------------------------------------------------------------------
        #                       Construct network components
        #
        # In this code, we use 'E_tie' to represent content encoder. 
        # 'E_src' and 'E_tar' are the appearance encoder for the different domain. 
        # --------------------------------------------------------------------------
        self.E_tie = Encoder(content_dims)
        self.E_src = Encoder(appearance_dims)
        self.E_tar = Encoder(appearance_dims)
        self.G_src = Generator(content_dims + appearance_dims)
        self.G_tar = Generator(content_dims + appearance_dims)
        self.D_src = Discriminator()
        self.D_tar = Discriminator()

        # Initial weight
        self.E_tie.apply(weights_init_Xavier)
        self.E_src.apply(weights_init_Xavier)
        self.E_tar.apply(weights_init_Xavier)
        self.G_src.apply(weights_init_Xavier)
        self.G_tar.apply(weights_init_Xavier)
        self.D_src.apply(weights_init_Xavier)
        self.D_tar.apply(weights_init_Xavier)

        # Define criterion
        self.crit_mse = nn.L1Loss()
        self.crit_vgg = VGGLoss()
        self.crit_adv = nn.BCEWithLogitsLoss()
        self.crit_kld = KLLoss()

        # Define optimizer
        self.optim_1st_E = optim.Adam(itertools.chain(self.E_tie.parameters(), self.E_src.parameters()), lr = 0.0001, betas=(0.0, 0.9))
        self.optim_1st_G = optim.Adam(self.G_src.parameters(), lr = 0.0001, betas=(0.0, 0.9))
        self.optim_1st_D = optim.Adam(self.D_src.parameters(), lr = 0.0001, betas=(0.0, 0.9))
        self.optim_2nd_E = optim.Adam(self.E_tar.parameters(), lr = 0.0001, betas=(0.0, 0.9))
        self.optim_2nd_G = optim.Adam(self.G_tar.parameters(), lr = 0.0001, betas=(0.0, 0.9))
        self.optim_2nd_D = optim.Adam(self.D_tar.parameters(), lr = 0.0001, betas=(0.0, 0.9))

        # Define scheduler
        self.scheduler_1st_E = optim.lr_scheduler.ExponentialLR(self.optim_1st_E, gamma = 0.99)
        self.scheduler_1st_G = optim.lr_scheduler.ExponentialLR(self.optim_1st_G, gamma = 0.99)
        self.scheduler_1st_D = optim.lr_scheduler.ExponentialLR(self.optim_1st_D, gamma = 0.99)
        self.scheduler_2nd_E = optim.lr_scheduler.ExponentialLR(self.optim_2nd_E, gamma = 0.99)
        self.scheduler_2nd_D = optim.lr_scheduler.ExponentialLR(self.optim_2nd_D, gamma = 0.99)
        self.to('cuda')

    ###########################################################################################################
    # IO
    ###########################################################################################################
    def load(self, path, stage = 1):
        """
            Load the pre-trained model

            Arg:    path    (Str)   - The path of pre-trained model
                    stage   (Int)   - The index of training stage
        """
        if os.path.exists(path):
            print("Load the pre-trained model from {}".format(path))
            state = torch.load(path)
            for (key, obj) in state.items():
                if len(key) > 10:
                    if key[1:9] == 'oss_list':
                        setattr(self, key, obj)
            self.E_tie.load_state_dict(state['E_tie'])
            self.E_src.load_state_dict(state['E_src'])
            self.G_src.load_state_dict(state['G_src'])
            self.D_src.load_state_dict(state['D_src'])
            if stage > 1:
                self.E_tar.load_state_dict(state['E_tar'])
                self.G_tar.load_state_dict(state['G_tar'])
                self.D_tar.load_state_dict(state['D_tar'])
        else:
            print("Pre-trained model {} is not exist...".format(path))

    def save(self, path):
        """
            Save the model parameters to the given path

            Arg:    path    (Str)   - The path of model where you want to save in
        """
        state = {
            'E_tie': self.E_tie.state_dict(),
            'E_src': self.E_src.state_dict(),
            'E_tar': self.E_tar.state_dict(),
            'G_src': self.G_src.state_dict(),
            'G_tar': self.G_tar.state_dict(),
            'D_src': self.D_src.state_dict(),
            'D_tar': self.D_tar.state_dict(),
        }
        for key in self.__dict__:
            if len(key) > 10:
                if key[1:9] == 'oss_list':
                    state[key] = getattr(self, key)
        torch.save(state, path)
        
    def copy(self):
        """
            Transfer the knowledge to the target network components
            You should call this function before you fine-tune the target generator!
        """
        self.E_tar.load_state_dict(self.E_src.state_dict())
        self.G_tar.load_state_dict(self.G_src.state_dict())
        self.D_tar.load_state_dict(self.D_src.state_dict())

    def fix(self):
        """
            Fix the source generator and content encoder without revising the parameters
            You should call this function before you fine-tune the target generator!
        """
        for m in self.E_tie.parameters():
            m.requires_grad = False
        for m in self.E_src.parameters():
            m.requires_grad = False
        for m in self.G_src.parameters():
            m.requires_grad = False
        for m in self.D_src.parameters():
            m.requires_grad = False

    ###########################################################################################################
    # Set and get
    ###########################################################################################################
    def setInput_1st(self, unpair_src_img, pair_src_img, pair_tar_img):
        """
            Set the input tensor for the 1st stage training
            You should call this function before you update the parameters

            Arg:    unpair_src_img  (torch.FloatTensor)     - The tensor of unpaired source image
                    pair_src_img    (torch.FloatTensor)     - The source image tensor with pair relationship
                    pair_tar_img    (torch.FloatTensor)     - The target image tensor with pair relationship
        """
        self.unpair_src_img = unpair_src_img.to('cuda')
        self.pair_src_img = pair_src_img.to('cuda')
        self.pair_tar_img = pair_tar_img.to('cuda')

    def setInput_2nd(self, pair_src_img, pair_tar_img):
        """
            Set the input tensor for the 2nd stage training
            You should call this function before you update the parameters

            Arg:    pair_src_img    (torch.FloatTensor)     - The source image tensor with pair relationship
                    pair_tar_img    (torch.FloatTensor)     - The target image tensor with pair relationship
        """
        self.pair_src_img  = pair_src_img.to('cuda')
        self.pair_tar_img  = pair_tar_img.to('cuda')

    def getLoss(self, stage = 1, normalize = False):
        """
            Return the latest loss value for the given stage number

            Arg:    stage       (Int)   - The index of training stage
                    normalize   (Bool)  - If get the average of the loss list or not
            Ret:    The dict object contain the pair of loss term name and the loss value
        """
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 13 and key[0] == 'l':
                if (stage == 1 and key[10:13] == '1st') or (stage == 2 and key[10:13] == '2nd'):
                    if not normalize:
                        loss_dict[key] = round(getattr(self, key)[-1], 6)
                    else:
                        loss_dict[key] = np.mean(getattr(self, key))
        return loss_dict

    def getLossList(self, stage = 1):
        """
            Return the loss list of each epoch for the given stage number

            Arg:    stage       (Int)   - The index of training stage
            Ret:    The dict object contain the pair of loss term name and the loss list
        """
        loss_dict = {}
        for key in self.__dict__:
            if len(key) > 13 and key[0] == 'L':
                if (stage == 1 and key[10:13] == '1st') or (stage == 2 and key[10:13] == '2nd'):
                    loss_dict[key] = getattr(self, key)
        return loss_dict

    ###########################################################################################################
    #   Sample function
    ###########################################################################################################
    def sampleN(self, batch_size):
        """
            Random sample the data from the Gaussian and generate the image for given several time

            Arg:    batch_size  (Int)   - The number of image you want to generate
            Ret:    The generated source image and target image
        """
        z = torch.cat([
            RANDN(torch.zeros([batch_size, self.content_dims, 1, 1]).cuda(), torch.ones([batch_size, self.content_dims, 1, 1]).cuda()),
            RANDN(torch.zeros([batch_size, self.appearance_dims, 1, 1]).cuda(), torch.ones([batch_size, self.appearance_dims, 1, 1]).cuda()),            
        ], 1)
        fake_src = self.G_src(z)
        fake_tar = self.G_tar(z)
        return fake_src, fake_tar

    ###########################################################################################################
    #   Forward function
    ###########################################################################################################
    def forward_1st(self):
        """
            1st forward procedure of LaDo
        """
        # Unpair data
        self.content_mean_src_1_unpair   , self.content_logvar_src_1_unpair    = self.E_tie(self.unpair_src_img)
        self.appearance_mean_src_1_unpair, self.appearance_logvar_src_1_unpair = self.E_src(self.unpair_src_img)
        self.content_z_src_1_unpair    = RANDN(self.content_mean_src_1_unpair   , self.content_logvar_src_1_unpair   )
        self.appearance_z_src_1_unpair = RANDN(self.appearance_mean_src_1_unpair, self.appearance_logvar_src_1_unpair)
        self.z_src_1_unpair = torch.cat([self.content_z_src_1_unpair, self.appearance_z_src_1_unpair], 1)
        self.fake_src_recon = self.G_src(self.z_src_1_unpair)
        self.content_mean_src_2_unpair, _    = self.E_tie(self.fake_src_recon)
        self.appearance_mean_src_2_unpair, _ = self.E_src(self.fake_src_recon)

        # Noise data
        self.content_z_src_1_noise    = RANDN(torch.zeros([self.batch_size, self.content_dims   , 1, 1]).cuda(), torch.ones([self.batch_size, self.content_dims   , 1, 1]).cuda())
        self.appearance_z_src_1_noise = RANDN(torch.zeros([self.batch_size, self.appearance_dims, 1, 1]).cuda(), torch.ones([self.batch_size, self.appearance_dims, 1, 1]).cuda())
        self.z_src_1_noise = torch.cat([self.content_z_src_1_noise, self.appearance_z_src_1_noise], 1)
        self.fake_src_noise = self.G_src(self.z_src_1_noise)

        # Pair data
        self.content_mean_src_1_pair, _ = self.E_tie(self.pair_src_img)
        self.content_mean_tar_1_pair, _ = self.E_tie(self.pair_tar_img)

    def forward_2nd(self):
        """
            2nd forward procedure of LaDo
        """
        # Pair forward
        self.content_mean_src_1_pair   , self.content_logvar_src_1_pair    = self.E_tie(self.pair_src_img)
        self.appearance_mean_tar_1_pair, self.appearance_logvar_tar_1_pair = self.E_tar(self.pair_tar_img)
        content_z_src_1_pair    = RANDN(self.content_mean_src_1_pair   , self.content_logvar_src_1_pair   )
        appearance_z_tar_1_pair = RANDN(self.appearance_mean_tar_1_pair, self.appearance_logvar_tar_1_pair)
        z_tar_1_pair = torch.cat([content_z_src_1_pair, appearance_z_tar_1_pair], 1)
        self.fake_tar_recon_cross = self.G_tar(z_tar_1_pair)

        # Target only
        content_mean_tar_1_straight, content_logvar_tar_1_straight = self.E_tie(self.pair_tar_img)
        content_z_tar_1_straight = RANDN(content_mean_tar_1_straight, content_logvar_tar_1_straight)
        z_tar_1_straight = torch.cat([content_z_tar_1_straight, appearance_z_tar_1_pair], 1)
        self.fake_tar_recon_straight = self.G_tar(z_tar_1_straight)

    ###########################################################################################################
    #   Backward for 1st step
    ###########################################################################################################
    def backward_1st(self):
        """
            Update the parameters for the LaDo 1st stage training
            We update the discriminator, generator and encoder in order
        """
        # Update for discriminator
        self.forward_1st()
        self.optim_1st_D.zero_grad()
        self.backward_1st_D()
        self.optim_1st_D.step()

        # Update for generator
        self.optim_1st_G.zero_grad()
        self.backward_1st_G()
        self.optim_1st_G.step()

        # Update for encoder
        self.forward_1st()
        self.optim_1st_E.zero_grad()
        self.backward_1st_E()
        self.optim_1st_E.step() 

    def backward_1st_D(self):
        """
            Update the discriminator for the LaDo 1st stage training
            The loss term only contains Image Adversarial loss (discriminator part)
        """
        real_logit  = self.D_src(self.unpair_src_img)
        fake_logit  = self.D_src(self.fake_src_recon.detach())
        noise_logit = self.D_src(self.fake_src_noise.detach())
        real_target = torch.ones(real_logit.size(0)).cuda()
        fake_target1 = torch.zeros(fake_logit.size(0)).cuda()
        fake_target2 = torch.zeros(noise_logit.size(0)).cuda()
        d_loss = self.crit_adv(real_logit, real_target) + (self.crit_adv(fake_logit, fake_target1) + self.crit_adv(noise_logit, fake_target2) ) / 2
        d_loss *= self.lambda_loss_1st_ia
        self.loss_list_1st_ia_d.append(d_loss.item())
        d_loss.backward()

    def backward_1st_G(self):
        """
            Update the generator for the LaDo 1st stage training
            The loss terms include:
                1. Image Reconstruction loss
                2. Image Adversarial loss (generator part)
                3. Latent Regression loss
        """
        # Image Reconstruction loss
        loss_ir = self.crit_mse(self.fake_src_recon, self.unpair_src_img) * self.lambda_loss_1st_ir
        self.loss_list_1st_ir.append(loss_ir.item())

        # Image Adversarial loss
        recon_logit = self.D_src(self.fake_src_recon)
        noise_logit = self.D_src(self.fake_src_noise)
        real_target1 = torch.ones(recon_logit.size(0)).cuda()
        real_target2 = torch.ones(noise_logit.size(0)).cuda()
        loss_ia = (self.crit_adv(recon_logit, real_target1) + self.crit_adv(noise_logit, real_target2)) / 2 
        loss_ia *= self.lambda_loss_1st_ia
        self.loss_list_1st_ia_g.append(loss_ia.item())

        # Latent Regression loss
        loss_content_lr    = self.crit_mse(self.content_mean_src_2_unpair   , self.content_z_src_1_unpair   )
        loss_appearance_lr = self.crit_mse(self.appearance_mean_src_2_unpair, self.appearance_z_src_1_unpair)
        loss_lr = (loss_content_lr + loss_appearance_lr) * self.lambda_loss_1st_lr
        self.loss_list_1st_lr.append(loss_lr.item())

        # Merge the loss and backward
        g_loss = loss_ir + loss_ia + loss_lr
        g_loss.backward()

    def backward_1st_E(self):
        """
            Update the encoder for the LaDo 1st stage training
            The loss terms include:
                1. Image Reconstruction loss
                2. KL loss
                3. Latent Regression loss
                4. Pair Structure loss
        """
        # Image Reconstruction loss
        loss_ir = self.crit_mse(self.fake_src_recon, self.unpair_src_img) * self.lambda_loss_1st_ir
        self.loss_list_1st_ir[-1] += loss_ir.item()

        # KL-divergence for both content and appearance
        self.loss_kld_src_content    = self.crit_kld(self.content_mean_src_1_unpair   , self.content_logvar_src_1_unpair   )
        self.loss_kld_src_appearance = self.crit_kld(self.appearance_mean_src_1_unpair, self.appearance_logvar_src_1_unpair)
        loss_kl = (self.loss_kld_src_content + self.loss_kld_src_appearance) * self.lambda_loss_1st_kl
        self.loss_list_1st_kl.append(loss_kl.item())        

        # Latent Regression loss
        loss_content_lr    = self.crit_mse(self.content_mean_src_2_unpair   , self.content_z_src_1_unpair   )
        loss_appearance_lr = self.crit_mse(self.appearance_mean_src_2_unpair, self.appearance_z_src_1_unpair)
        loss_lr = (loss_content_lr + loss_appearance_lr) * self.lambda_loss_1st_lr
        self.loss_list_1st_lr[-1] += loss_lr.item()

        # Pair Structure loss
        loss_ps = self.crit_mse(self.content_mean_tar_1_pair, self.content_mean_src_1_pair) * self.lambda_loss_1st_ps
        self.loss_list_1st_ps.append(loss_ps.item())        

        # Backward the total loss
        e_loss = loss_ir + loss_kl + loss_lr + loss_ps
        e_loss.backward()

    ###########################################################################################################
    #   Backward for 2nd step
    ###########################################################################################################
    def backward_2nd(self):
        """
            Update the parameters for the LaDo 2nd stage training
            We update the discriminator, generator and encoder in order
        """
        # Update for discriminator
        self.forward_2nd()
        self.optim_2nd_D.zero_grad()
        self.backward_2nd_D()
        self.optim_2nd_D.step()

        # Update for generator
        self.optim_2nd_G.zero_grad()
        self.backward_2nd_G()
        self.optim_2nd_G.step()

        # Update for encoder
        self.forward_2nd()
        self.optim_2nd_E.zero_grad()
        self.backward_2nd_E()
        self.optim_2nd_E.step() 

    def backward_2nd_D(self):
        """
            Update the discriminator for the LaDo 2nd stage training
            The loss term only contains Image Adversarial loss (discriminator part)
        """
        real_logit  = self.D_tar(self.pair_tar_img).squeeze()
        fake_logit1 = self.D_tar(self.fake_tar_recon_cross.detach()).squeeze()
        fake_logit2 = self.D_tar(self.fake_tar_recon_straight.detach()).squeeze()
        d_loss = (self.crit_adv(real_logit, torch.ones(real_logit.size(0)).cuda()) + \
            (self.crit_adv(fake_logit1, torch.zeros(fake_logit1.size(0)).cuda()) + \
             self.crit_adv(fake_logit2, torch.zeros(fake_logit2.size(0)).cuda())) / 2)
        d_loss *= self.lambda_loss_2nd_ia
        self.loss_list_2nd_ia_d.append(d_loss.item())
        d_loss.backward()

    def backward_2nd_G(self):
        """
            Update the generator for the LaDo 1st stage training
            The loss terms include:
                1. Image Reconstruction loss
                2. Image Adversarial loss (generator part)
                3. Pair Generation loss
        """
        # Image Reconstruction loss
        loss_ir = self.crit_vgg(self.fake_tar_recon_straight, self.pair_tar_img) * self.lambda_loss_2nd_ir
        self.loss_list_2nd_ir.append(loss_ir.item())

        # Image Adversarial loss
        fake_logit1 = self.D_tar(self.fake_tar_recon_cross).squeeze()
        fake_logit2 = self.D_tar(self.fake_tar_recon_straight).squeeze()
        loss_ia = ( self.crit_adv(fake_logit1, torch.ones(fake_logit1.size(0)).cuda()) + \
                        self.crit_adv(fake_logit2, torch.ones(fake_logit2.size(0)).cuda()) \
        ) / 2 
        loss_ia *= self.lambda_loss_2nd_ia
        self.loss_list_2nd_ia_g.append(loss_ia.item())

        # Pair Generation loss
        loss_pg = self.crit_vgg(self.fake_tar_recon_cross, self.pair_tar_img) * self.lambda_loss_2nd_pg
        self.loss_list_2nd_pg.append(loss_pg.item())

        # Merge the loss and backward
        g_loss = loss_ir + loss_ia + loss_pg
        g_loss.backward()

    def backward_2nd_E(self):
        """
            Update the encoder for the LaDo 1st stage training
            The loss terms include:
                1. Image Reconstruction loss
                2. KL loss
                3. Pair Generation loss
        """
        # Image Reconstruction loss
        loss_ir = self.crit_vgg(self.fake_tar_recon_straight, self.pair_tar_img) * self.lambda_loss_2nd_ir
        self.loss_list_2nd_ir[-1] += loss_ir.item()

        # KL-divergence for appearance
        loss_kl = self.crit_kld(self.appearance_mean_tar_1_pair, self.appearance_logvar_tar_1_pair) * self.lambda_loss_2nd_kl
        self.loss_list_2nd_kl.append(loss_kl.item())

        # Pair Generation loss
        loss_pg = self.crit_vgg(self.fake_tar_recon_cross, self.pair_tar_img) * self.lambda_loss_2nd_pg
        self.loss_list_2nd_pg[-1] += loss_pg.item()

        # Merge the loss and backward
        e_loss = loss_ir + loss_kl + loss_pg
        e_loss.backward()

    ###########################################################################################################
    #   Finish training
    ###########################################################################################################
    def finishEpoch(self, stage = 1):
        """
            Apply the learning decay and summary the loss list

            Arg:    stage       (Int)   - The index of training stage
        """
        for key in self.__dict__:
            if len(key) > 13 and key[0] == 'l':
                list_obj = getattr(self, key)
                if isinstance(list_obj, list) and len(list_obj) > 0:
                    List_obj = getattr(self, 'L' + key[1:])
                    List_obj.append(np.mean(list_obj))
                    setattr(self, key, [])
                    setattr(self, 'L' + key[1:], List_obj)
        if stage == 1:
            self.scheduler_1st_E.step()
            self.scheduler_1st_G.step()
            self.scheduler_1st_D.step()
        elif stage == 2:
            self.scheduler_2nd_E.step()
            self.scheduler_2nd_G.step()
            self.scheduler_2nd_D.step()
        else:
            raise Exception()