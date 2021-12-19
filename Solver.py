# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import os
import torch
import torch.optim as optim
from read_data import MyData
from torch.optim.lr_scheduler import LinearLR
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from datetime import datetime
from loss import cal_loss,SSIMLoss, CompoundLoss, gradient_penalty, bayeGen_loss
from utils import output2complex,gray2rgb
from os.path import join
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import normalized_root_mse as cal_nrmse
import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import time
from model.DCCNN import DnCn,Hybrid_DnCn
from model.LPDNET import LPDNet #,Origin_LPDNet #LPDNet_Unet
from model.RefineGAN import Refine_G, Discriminator, Refine_G_D5C5, Refine_G_LPD
from model.MICCAN import MICCANlong
import torch.nn as nn
from tqdm import tqdm
from model.Dense_LPDNET import Dense_LPDNet
from loss_external import SpatialCorrelativeLoss
import loss_external
import h5py

class Solver():
    def __init__(self, args):
        self.args = args
        ################  experiment settings  ################
        self.expname = self.args.expname
        self.data_name = self.args.dataset
        self.model_name = self.args.model
        self.mask_name = self.args.mask_type  # mask type
        self.acc = self.args.acc  # for ocmr dataset only
        self.sampling_rate = self.args.sampling_rate  # sampling rate, 10, 20 ,30 ,40 ...
        self.imageDir_train = self.args.train_path  # train path
        self.imageDir_val = self.args.val_path  # val path while training
        self.num_epoch = self.args.num_epoch
        self.batch_size = self.args.batch_size  # batch size
        self.val_on_epochs = self.args.val_on_epochs # val on each val_on_epochs epochs;
        self.data_aug = self.args.data_aug
        self.weight_cliping_limit = 0.01 # for gan only
        ## settings for optimizer
        self.lr = self.args.lr
        self.loss_name = self.args.loss
        self.end_factor = self.args.end_factor
        self.beta = self.args.beta
        ## settings for data preprocessing
        self.img_size = (192,160) if self.data_name == 'ocmr' else (256,256)
        self.norm_type = self.args.norm_type
        self.gen2_flag = 'refine' in self.model_name  ## whether to generate 2 img , for refinegan, refine_d5c5
        self.maskDir = 'data/mask/' + self.mask_name + '/mask_' + str(
            self.sampling_rate // 10) + '/'  # different sampling rate
        self.saveDir = 'weight'  # model save path while training
        if not os.path.isdir(self.saveDir):
            os.makedirs(self.saveDir)

        self.imageDir_test = self.args.test_path  # test path

        self.task_name = self.model_name + '_' + self.data_name + '_' + self.mask_name + '_' + str(self.sampling_rate)\
                         + '_lr_' + str(np.log10(self.lr)) +'_end_factor_' + str(self.end_factor) + '_beta_' + str(self.beta)\
                         + '_norm_type_' + str(self.norm_type) + '_bs_' + str(self.batch_size)+'_loss_'+self.loss_name\
                         +'_aug_'+str(self.data_aug)+'_'+self.expname

        self.model_path = 'weight/' + self.task_name + '_' + 'best.pth'   # model load path for test and visualization
        ###################################################################################
        print('task_name: ', self.task_name)
        
        #### temperate add
        # self.netPre = loss_external.VGG16().cuda()
        # self.criterionSpatial = SpatialCorrelativeLoss()
        # self.vgg_norm = loss_external.Normalization()

    # def Spatial_Loss(self, net, src, tgt, other=None):
    #     """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
    #     attn_layers = [4, 7, 9]
    #     n_layers = len(attn_layers)
    #     feats_src = net(src, attn_layers, encode_only=True)
    #     feats_tgt = net(tgt, attn_layers, encode_only=True)
    #     if other is not None:
    #         feats_oth = net(torch.flip(other, [2, 3]), attn_layers, encode_only=True)
    #     else:
    #         feats_oth = [None for _ in range(n_layers)]
    #
    #     total_loss = 0.0
    #     for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
    #         loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
    #         total_loss += loss.mean()
    #
    #     if not self.criterionSpatial.conv_init:
    #         self.criterionSpatial.update_init_()
    #
    #     return total_loss / n_layers

    def train(self):

        ############################################ Specify network ############################################
        if self.model_name == 'd5c5':
            self.G = DnCn(2, 5, 5)
        elif self.model_name == 'hybrid':
            self.G = Hybrid_DnCn()
        elif self.model_name == 'refine_G':
            self.G = Refine_G()
        elif self.model_name == 'refine_lpd':
            self.G = Refine_G_LPD()
        elif self.model_name == 'refinegan':
            self.G = Refine_G()
            self.D = Discriminator()
        elif self.model_name == 'refinegan_d5c5':
            self.G = Refine_G_D5C5()
            self.D = Discriminator()
        elif self.model_name == 'refinegan_lpd':
            self.G = Refine_G_LPD()
            self.D = Discriminator()
        elif self.model_name == 'refinegan_lpd_im':
            self.G = Refine_G_LPD()
            self.D = Discriminator()
        elif self.model_name == 'pdnet':
            self.G = LPDNet() #
        elif self.model_name == 'dense_pdnet':
            self.G = Dense_LPDNet(growth_rate=10, block_config=(2, 4, 4), compression=0.5,
                 num_init_features=10, bn_size=32, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=True)
        elif self.model_name =='miccan':
            self.G = MICCANlong(2, 2, 5, block='UCA')
        print('    Total params: %.5fMB' % (sum(p.numel() for p in self.G.parameters()) / (1024.0 * 1024)))
        # print(self.G)
        ############################################ Specify loss ############################################
        if self.loss_name == 'mse':  # or self.model_name == 'd5c5':
            self.criterion = nn.MSELoss()
        elif self.loss_name == 'compound':
            self.criterion = CompoundLoss()
        elif self.loss_name == 'ssim':
            self.criterion = SSIMLoss().cuda()
        elif self.loss_name == 'l1':
            self.criterion = nn.L1Loss()
        # elif self.loss_name == 'compound2':
        #     self.criterion = self.Spatial_Loss
    ############################################ Specify optimizer ############################################
        if self.beta:
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-10)
            if 'gan' in self.model_name :
                self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999), eps=1e-3, weight_decay=1e-10)
        else:
            self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-10)
            # self.optimizer_G_2 = optim.Adam([self.G.bb], lr=self.lr, eps=1e-2, weight_decay=1e-10)
            if 'gan' in self.model_name :
                self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-10)
        self.scheduler_lr_G = LinearLR(self.optimizer_G, start_factor=1., end_factor=self.end_factor, total_iters=self.num_epoch)
        if 'gan' in self.model_name :
            self.scheduler_lr_D = LinearLR(self.optimizer_D, start_factor=1., end_factor=self.end_factor, total_iters=self.num_epoch)
        self.G.cuda()
        if 'gan' in self.model_name :
            self.D.cuda()
        ############################################ load data ############################################

        dataset_train = MyData(self.imageDir_train, self.maskDir, self.sampling_rate, self.img_size, is_training='train', \
                               dataset=self.data_name,gen2=self.gen2_flag,norm = self.norm_type)
        dataset_val = MyData(self.imageDir_val, self.maskDir, self.sampling_rate,self.img_size, is_training='val', \
                             dataset=self.data_name,gen2=self.gen2_flag,norm = self.norm_type)

        loader_train = Data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=4, pin_memory=True)
        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)
        self.slices_val = len(dataset_val)
        print("slices of 2d train data: ", len(dataset_train))
        print("slices of 2d validation data: ", len(dataset_val))

        ############################################ setting for tensorboard ############################################
        self.TIMESTAMP = self.task_name  #+ "_{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.writer = SummaryWriter('log/' + self.TIMESTAMP)

        ############################################ start to run epochs ############################################

        start_epoch = 0
        best_val_psnr = 0
        for epoch in range(start_epoch, self.num_epoch):
            ####################### 1. training #######################
            if 'gan' in self.model_name :
                if 'im' in self.model_name :
                    loss_g = self._train_gan_2(loader_train,epoch)
                else:
                    loss_g = self._train_gan(loader_train,epoch)
            else:
                loss_g = self._train_cnn(loader_train)
            ####################### 2. validate #######################
            if epoch % self.val_on_epochs == 0:
                base_psnr,test_psnr,base_ssim,test_ssim = self._validate(loader_val)

                ########################## 3. print and tensorboard ########################
                print("Epoch {}/{}".format(epoch + 1, self.num_epoch))
                print(" base PSNR:\t\t{:.6f}".format(base_psnr))
                print(" test PSNR:\t\t{:.6f}".format(test_psnr))
                print(" base SSIM:\t\t{:.6f}".format(base_ssim))
                print(" test SSIM:\t\t{:.6f}".format(test_ssim))
                ## write to tensorboard
                self.writer.add_scalar("loss/G_loss_total", loss_g, epoch)
                self.writer.add_scalar("loss/base_psnr", base_psnr, epoch)
                self.writer.add_scalar("loss/test_psnr", test_psnr, epoch)
                self.writer.add_scalar("loss/base_ssim", base_ssim, epoch)
                self.writer.add_scalar("loss/test_ssim", test_ssim, epoch)
                ## save the best model according to validation psnr
                if best_val_psnr < test_psnr:
                    best_val_psnr = test_psnr
                    best_name = self.TIMESTAMP + '_' + 'best.pth'   ###
                    state = {'net': self.G.state_dict(), 'start_epoch': epoch, 'psnr': test_psnr}
                    torch.save(state, join(self.saveDir, best_name))
        self.writer.close()

    def _train_gan(self,loader_train,epoch):
        ####################### 1. training #######################
        self.G.train()
        for data_dict in tqdm(loader_train):
            im_A, im_A_und, k_A_und, im_B, im_B_und, k_B_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                'im_A_und'].float().cuda(), data_dict['k_A_und'].float().cuda(), \
                                                                     data_dict['im_B'].float().cuda(), data_dict[
                                                                         'im_B_und'].float().cuda(), data_dict[
                                                                         'k_B_und'].float().cuda(), data_dict[
                                                                         'mask_A'].float().cuda()
            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            Sp1, S1, Tp1, T1 = self.G(im_A_und, k_A_und, mask)
            Sp2, S2, Tp2, T2 = self.G(im_B_und, k_B_und, mask)
            S1_dis_real = self.D(im_A)
            S2_dis_real = self.D(im_B)

            ############################################# 1.1 update discriminator #############################################
            with torch.no_grad():
                S1_clone = S1.detach().clone()
                T1_clone = T1.detach().clone()
                S2_clone = S2.detach().clone()
                T2_clone = T2.detach().clone()

            S1_dis_fake = self.D(S1_clone)
            T1_dis_fake = self.D(T1_clone)
            S2_dis_fake = self.D(S2_clone)
            T2_dis_fake = self.D(T2_clone)

            loss_dd, d_loss_list = cal_loss(im_A, k_A_und, im_B, k_B_und, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2,
                                           S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake,
                                           T2_dis_fake, cal_G=False)
            # wgan-gp: calculate graident penalty
            alpha = torch.rand(im_A_und.size(0), 1, 1, 1).cuda()
            x_hat = (alpha * im_A_und.data + (1 - alpha) * T1.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)

            loss_d = loss_dd + 50 * d_loss_gp

            self.optimizer_D.zero_grad()
            loss_d.backward()
            self.optimizer_D.step()

            # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit , for WGAN
            # for p in self.D.parameters():
            #     p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

            ############################################# 1.2 update generator #############################################
            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update

            S1_dis_fake = self.D(S1)
            T1_dis_fake = self.D(T1)
            S2_dis_fake = self.D(S2)
            T2_dis_fake = self.D(T2)
            loss_g, g_loss_list = cal_loss(im_A, k_A_und, im_B, k_B_und, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2,
                                           S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake,
                                           T2_dis_fake, cal_G=True)
            self.optimizer_G.zero_grad()
            loss_g.backward()
            self.optimizer_G.step()
        self.scheduler_lr_G.step()
        self.scheduler_lr_D.step()
        self.writer.add_scalar("loss/D_loss_total", loss_d, epoch)
        self.writer.add_scalar("loss/G_loss_AA", g_loss_list[0], epoch)
        self.writer.add_scalar("loss/G_loss_Aa", g_loss_list[1], epoch)
        self.writer.add_scalar("loss/recon_img_AA", g_loss_list[2], epoch)
        self.writer.add_scalar("loss/recon_img_Aa", g_loss_list[3], epoch)
        self.writer.add_scalar("loss/error_img_AA", g_loss_list[4], epoch)
        self.writer.add_scalar("loss/error_img_Aa", g_loss_list[5], epoch)
        self.writer.add_scalar("loss/recon_frq_AA", g_loss_list[6], epoch)
        self.writer.add_scalar("loss/recon_frq_Aa", g_loss_list[7], epoch)
        self.writer.add_scalar("loss/smoothness_AA", g_loss_list[8], epoch)
        self.writer.add_scalar("loss/smoothness_Aa", g_loss_list[9], epoch)
        self.writer.add_scalar("loss/D_loss_AA", d_loss_list[0], epoch)
        self.writer.add_scalar("loss/D_loss_Aa", d_loss_list[1], epoch)
        self.writer.add_scalar("loss/D_loss_AB", d_loss_list[2], epoch)
        self.writer.add_scalar("loss/D_loss_Ab", d_loss_list[3], epoch)
        return loss_g

    def _train_gan_2(self,loader_train,epoch):
        ####################### 1. training #######################
        self.G.train()
        for data_dict in tqdm(loader_train):
            im_A, im_A_und, k_A_und, im_B, im_B_und, k_B_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                'im_A_und'].float().cuda(), data_dict['k_A_und'].float().cuda(), \
                                                                     data_dict['im_B'].float().cuda(), data_dict[
                                                                         'im_B_und'].float().cuda(), data_dict[
                                                                         'k_B_und'].float().cuda(), data_dict[
                                                                         'mask_A'].float().cuda()
            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            T1, out_alpha1, out_beta1,_ = self.G(im_A_und, k_A_und, mask)
            T2, out_alpha2,out_beta2,_ = self.G(im_B_und, k_B_und, mask)


            T1 = output2complex(T1, revert=self.norm_type).unsqueeze(1)
            T2 = output2complex(T2, revert=self.norm_type).unsqueeze(1)
            im_A = output2complex(im_A, revert=self.norm_type).unsqueeze(1)
            im_B = output2complex(im_B, revert=self.norm_type).unsqueeze(1)
            im_A_und = output2complex(im_A_und, revert=self.norm_type).unsqueeze(1)
            im_B_und = output2complex(im_B_und, revert=self.norm_type).unsqueeze(1)
            # pred_dc = output2complex(pred_dc, revert=self.norm_type)
            ############################################# 1.2 update generator #############################################


            S1_dis_real = self.D(im_A)
            S2_dis_real = self.D(im_B)

            ############################################# 1.1 update discriminator #############################################
            with torch.no_grad():

                T1_clone = T1.detach().clone()
                T2_clone = T2.detach().clone()

            S1_dis_fake = None #self.D(S1_clone)
            T1_dis_fake = self.D(T1_clone)
            S2_dis_fake = None #self.D(S2_clone)
            T2_dis_fake = self.D(T2_clone)

            loss_dd, d_loss_list = cal_loss(im_A, k_A_und, im_B, k_B_und, mask,  T1, T2,
                                           S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake,
                                           T2_dis_fake, cal_G=False)
            # wgan-gp: calculate graident penalty
            alpha = torch.rand(im_A.size(0), 1, 1, 1).cuda()
            x_hat = (alpha * im_A_und.data + (1 - alpha) * T1.data).requires_grad_(True)
            out_src = self.D(x_hat)
            d_loss_gp = gradient_penalty(out_src, x_hat)

            alpha2 = torch.rand(im_B.size(0), 1, 1, 1).cuda()
            x_hat2 = (alpha2 * im_B_und.data + (1 - alpha2) * T2.data).requires_grad_(True)
            out_src2 = self.D(x_hat2)
            d_loss_gp = d_loss_gp + gradient_penalty(out_src2, x_hat2)


            loss_d = loss_dd + 50 * d_loss_gp
            self.optimizer_D.zero_grad()
            loss_d.backward()
            self.optimizer_D.step()

            # Clamp parameters to a range [-c, c], c=self.weight_cliping_limit , for WGAN
            # for p in self.D.parameters():
            #     p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

            ############################################# 1.2 update generator #############################################
            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = False  # they are set to False below in netG update

            S1_dis_fake = None #self.D(S1)
            T1_dis_fake = self.D(T1)
            S2_dis_fake = None #self.D(S2)
            T2_dis_fake = self.D(T2)

            # loss_bayes = bayeGen_loss(T1,out_alpha1,out_beta1,im_A) * 0.01
            # loss_bayes = loss_bayes + bayeGen_loss(T2, out_alpha2, out_beta2, im_B) * 0.01

            loss_g, g_loss_list = cal_loss(im_A, k_A_und, im_B, k_B_und, mask, T1,  T2,
                                           S1_dis_real, S1_dis_fake, T1_dis_fake, S2_dis_real, S2_dis_fake,
                                           T2_dis_fake, cal_G=True)
            # loss_g = loss_g #+ loss_bayes
            self.optimizer_G.zero_grad()
            loss_g.backward()
            self.optimizer_G.step()
        self.scheduler_lr_G.step()
        self.scheduler_lr_D.step()
        self.writer.add_scalar("loss/D_loss_total", loss_d, epoch)
        self.writer.add_scalar("loss/D_loss_gan", loss_dd, epoch)
        self.writer.add_scalar("loss/D_loss_gp", d_loss_gp, epoch)
        # self.writer.add_scalar("loss/G_loss_AA", g_loss_list[0], epoch)
        self.writer.add_scalar("loss/G_loss_Aa", g_loss_list[0], epoch)
        # self.writer.add_scalar("loss/recon_img_AA", g_loss_list[2], epoch)
        self.writer.add_scalar("loss/recon_img_Aa", g_loss_list[1], epoch)
        # self.writer.add_scalar("loss/error_img_AA", g_loss_list[4], epoch)
        # self.writer.add_scalar("loss/error_img_Aa", g_loss_list[5], epoch)
        # self.writer.add_scalar("loss/recon_frq_AA", g_loss_list[6], epoch)
        # self.writer.add_scalar("loss/recon_frq_Aa", g_loss_list[7], epoch)
        # self.writer.add_scalar("loss/smoothness_AA", g_loss_list[8], epoch)
        self.writer.add_scalar("loss/smoothness_Aa", g_loss_list[2], epoch)
        # self.writer.add_scalar("loss/D_loss_AA", d_loss_list[0], epoch)
        self.writer.add_scalar("loss/D_loss_Aa", d_loss_list[0], epoch)
        # self.writer.add_scalar("loss/D_loss_AB", d_loss_list[2], epoch)
        self.writer.add_scalar("loss/D_loss_Ab", d_loss_list[1], epoch)
        return loss_g

    def _train_cnn(self,loader_train):
        self.G.train()
        # history = None
        for data_dict in tqdm(loader_train):
            im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                'im_A_und'].float().cuda(), \
                                            data_dict['k_A_und'].float().cuda(), data_dict[
                                                'mask_A'].float().cuda()
            if 'refine' in self.model_name :
                Sp1, S1, Tp1, T1 = self.G(im_A_und, k_A_und, mask)
            else:
                T1 = self.G(im_A_und, k_A_und, mask) #,out_alpha,out_beta

            T1 = output2complex(T1, revert=self.norm_type)
            im_A = output2complex(im_A, revert=self.norm_type)
            # pred_dc = output2complex(pred_dc, revert=self.norm_type)
            ############################################# 1.2 update generator #############################################
            # loss_bayes =bayeGen_loss(T1,out_alpha,out_beta,im_A) * 0.01
            if self.loss_name == 'compound2':
                compound_loss = self.criterion(self.netPre,self.vgg_norm(gray2rgb(im_A)),self.vgg_norm(gray2rgb(T1))) #,gray2rgb(im_A_und),
            else:
                compound_loss = self.criterion(T1, im_A)
                # compound_loss_dc = self.criterion(pred_dc, im_A)*0.1
            loss_g = compound_loss #+  loss_bayes # + compound_loss_dc
            self.optimizer_G.zero_grad()
            loss_g.backward()
            self.optimizer_G.step()
        self.scheduler_lr_G.step()
        # print('loss_bayes: ', loss_bayes.item(), 'compound_loss: ',  compound_loss.item(),'loss_g: ', loss_g.item())
        #'compound_loss_dc: ',compound_loss_dc.item(),
        return loss_g

    def _validate(self,loader_val):

        base_psnr = 0
        test_psnr = 0
        base_ssim = 0
        test_ssim = 0

        self.G.eval()
        with torch.no_grad():
            for data_dict in tqdm(loader_val):

                im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                    'im_A_und'].float().cuda(), data_dict['k_A_und'].float().cuda(), data_dict[
                                                    'mask_A'].float().cuda()
                if 'gan' in self.model_name or 'refine' in self.model_name :
                    Sp1, S1, Tp1, T1 = self.G(im_A_und, k_A_und, mask)
                else:
                    # history = im_A_und
                    T1 = self.G(im_A_und, k_A_und, mask) #,out_alpha,out_beta

                ############## convert model ouput to complex value in original range
                T1 = output2complex(T1, revert=self.norm_type)
                im_A = output2complex(im_A, revert=self.norm_type)
                im_A_und = output2complex(im_A_und, revert=self.norm_type)

                ########################## 2.1 cal psnr for validation ###################################
                ########################### cal ssim ###################################
                for T1_i, im_A_i, im_A_und_i in zip(T1.cpu().numpy(), im_A.cpu().numpy(),
                                                    im_A_und.cpu().numpy()):
                    # print(im_A_i.dtype,im_A_i.min(),im_A_i.max())
                    ## for skimage.metrics, input is (im_true,im_pred)
                    base_ssim += cal_ssim(im_A_i, im_A_und_i)  # true, pred
                    test_ssim += cal_ssim(im_A_i, T1_i)
                    base_psnr += cal_psnr(im_A_i, im_A_und_i, data_range=im_A_i.max())
                    test_psnr += cal_psnr(im_A_i, T1_i, data_range=im_A_i.max())

        base_psnr /= self.slices_val
        test_psnr /= self.slices_val
        base_ssim /= self.slices_val
        test_ssim /= self.slices_val
        return base_psnr,test_psnr,base_ssim,test_ssim

    def test(self):

        ############################################ load data ################################

        dataset_val = MyData(self.imageDir_test, self.maskDir, self.sampling_rate, self.img_size, is_training='test',dataset=self.data_name,\
                             gen2=self.gen2_flag,norm = self.norm_type)


        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)
        len_data = len(dataset_val)
        print("slices of 2d test data: ", len_data)
        ####################### load model #######################
        if self.model_name == 'refinegan':
            G = Refine_G()
        elif self.model_name == 'hybrid':
            G = Hybrid_DnCn()
        elif self.model_name == 'd5c5':
            G = DnCn(2, 5, 5)
        elif self.model_name == 'refine_lpd':
            G = Refine_G_LPD()
        elif self.model_name == 'refine_G':
            G = Refine_G()
        elif self.model_name == 'refinegan_d5c5':
            G = Refine_G_D5C5()
        elif self.model_name == 'refinegan_lpd' or self.model_name == 'refinegan_lpd_im':
            G = Refine_G_LPD()
        elif self.model_name == 'pdnet':
            G = LPDNet()
        print('best epoch at :', torch.load(self.model_path)['start_epoch'])
        print('    Total params: %.5fMB' % (sum(p.numel() for p in G.parameters()) / (1024.0 * 1024)))
        G.load_state_dict(torch.load(self.model_path)['net'])
        G.cuda()
        G.eval()
        #######################validate
        base_psnr = 0
        test_psnr = 0
        base_ssim = 0
        test_ssim = 0
        base_nrmse = 0
        test_nrmse = 0
        with torch.no_grad():
            time_0 =time.time()
            for data_dict in tqdm(loader_val):
                im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict['im_A_und'].float().cuda(), \
                                                data_dict['k_A_und'].float().cuda(), \
                                                data_dict['mask_A'].float().cuda()
                if 'gan' in self.model_name :
                    Sp1, S1, Tp1, T1 = G(im_A_und, k_A_und, mask)
                else:
                    T1 = G(im_A_und, k_A_und, mask)
                ############## convert model ouput to complex value in original range

                T1 = output2complex(T1,revert = self.norm_type)
                im_A = output2complex(im_A,revert = self.norm_type)
                im_A_und = output2complex(im_A_und,revert = self.norm_type)
                #
                # ########################## cal psnr ###################################
                # ########################### cal ssim ###################################
                for T1_i, im_A_i, im_A_und_i in zip(T1.cpu().numpy(), im_A.cpu().numpy(), im_A_und.cpu().numpy()):
                    ## for skimage.metrics, input is (im_true,im_pred)
                    base_nrmse += cal_nrmse( im_A_i,im_A_und_i)
                    test_nrmse += cal_nrmse(im_A_i,T1_i)
                    base_ssim += cal_ssim(im_A_i, im_A_und_i)  # true, pred
                    test_ssim += cal_ssim(im_A_i, T1_i)
                    base_psnr += cal_psnr(im_A_i, im_A_und_i,data_range=im_A_i.max())
                    test_psnr += cal_psnr(im_A_i, T1_i,data_range=im_A_i.max())
                ########################### cal nmse ###################################
            time_1 = time.time()
            print('inference speed: {:.5f} us'.format(1000*(time_1-time_0)/len_data))
        base_psnr /= len_data
        test_psnr /= len_data
        base_ssim /= len_data
        test_ssim /= len_data
        base_nrmse /= len_data
        test_nrmse /= len_data

        print(" base PSNR:\t\t{:.6f}".format(base_psnr))
        print(" test PSNR:\t\t{:.6f}".format(test_psnr))
        print(" base SSIM:\t\t{:.6f}".format(base_ssim))
        print(" test SSIM:\t\t{:.6f}".format(test_ssim))
        print(" base NRMSE:\t\t{:.6f}".format(base_nrmse))
        print(" test NRMSE:\t\t{:.6f}".format(test_nrmse))


    def visualize(self):

        dataset_val = MyData(self.imageDir_test, self.maskDir, self.sampling_rate, self.img_size, is_training='test',dataset=self.data_name,\
                             gen2=self.gen2_flag,norm = self.norm_type)

        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)
        len_data = len(dataset_val)
        print("slices of 2d test data: ", len_data)
        ####################### load model #######################

        G0 = DnCn(2, 5, 5)
        G1 = Origin_LPDNet()
        G2 = LPDNet()
        G3 = LPDNet_Unet()

        if self.sampling_rate==20:
            p0 = 'weight/d5c5_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1__best.pth'
            p1 = 'weight/pdnet_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_lpd10-nonzero_init-compound_best.pth'
            p2 = 'weight/pdnet_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_pdnet-my3_residual_1_5xbuffer_compound-full_best.pth'
            p3 = 'weight/pdnet_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_unet-pdnet-my3_residual_1_5xbuffer_compound-full_best.pth'

        elif self.sampling_rate==10:
            p0 = 'weight/d5c5_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_d5c5_compound-full_best.pth'
            p1 = 'weight/pdnet_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_pdnet_5xbuffer_compound-full_best.pth'
            p2 = 'weight/pdnet_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_pdnet-my3_residual_1_5xbuffer_compound-full_best.pth'
            p3 = 'weight/pdnet_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_unet-pdnet-my3_residual_1_5xbuffer_compound-full_best.pth'

        G0.load_state_dict(torch.load(p0)['net'])
        G1.load_state_dict(torch.load(p1)['net'])
        G2.load_state_dict(torch.load(p2)['net'])
        G3.load_state_dict(torch.load(p3)['net'])
        G0.cuda()
        G0.eval()
        G1.cuda()
        G1.eval()
        G2.cuda()
        G2.eval()
        G3.cuda()
        G3.eval()

        ####################### evaluate #######################
        data_iter = iter(loader_val)
        for _ in range(780):   #### 20x: 780  # 10x: 50
            data_dict = next(data_iter)

        with torch.no_grad():
            im_A_, im_A_und_, k_A_und_, mask_ = data_dict['im_A'].float().cuda(), data_dict['im_A_und'].float().cuda(), \
                                            data_dict['k_A_und'].float().cuda(), \
                                            data_dict['mask_A'].float().cuda()
            # pred_list=[im_A_und_[0,0].cpu().numpy()]
            # error_list=[abs(output2complex(im_A_)[0].cpu().numpy()-output2complex(im_A_und_)[0,0].cpu().numpy())]
            visual_dir = 'visual/'
            if not os.path.isdir(visual_dir):
                os.makedirs(visual_dir)
            zf = output2complex(im_A_und_)[0].cpu().numpy()
            plt.imsave(visual_dir + 'zf.png', zf.clip(0,zf.max()*0.5) , cmap='gray')
            gt = output2complex(im_A_)[0].cpu().numpy()
            plt.imsave(visual_dir+'gt.png',gt.clip(0,gt.max()*0.5),cmap='gray')
            mm = np.fft.ifftshift(mask_[0,0].cpu().numpy(), axes=(-1, -2))
            plt.imsave(visual_dir + 'mask.png',mm ,cmap='gray')
            print(abs(output2complex(im_A_und_)[0]-output2complex(im_A_)[0]).cpu().numpy().max())
            plt.imsave(visual_dir + 'zf_error.png', abs(output2complex(im_A_und_)[0]-output2complex(im_A_)[0]).cpu().numpy(),vmax=0.5)
            for i,G in enumerate([G0,G1,G2,G3]):
                T1 = G(im_A_und_, k_A_und_, mask_)
                ############## convert model ouput to complex value in original range
                T1 = output2complex(T1).cpu().numpy()
                im_A = output2complex(im_A_).cpu().numpy()
                im_A_und = output2complex(im_A_und_).cpu().numpy()

                ########################## cal psnr ###################################
                base_psnr = cal_psnr(im_A, im_A_und,data_range=im_A.max())
                test_psnr = cal_psnr(im_A, T1,data_range=im_A.max())
                print('acc factor: ', str( 100 // self.sampling_rate))
                print(" base PSNR:\t\t{:.6f}".format(base_psnr))
                print(" test PSNR:\t\t{:.6f}".format(test_psnr))
                ######################### visualization ###############################


                T1 = T1[0]
                im_A = im_A[0]
                im_A_und = im_A_und[0]
                # error_und = abs(im_A_und - im_A)
                err_T1 = abs(T1 - im_A)

                plt.imsave(visual_dir + 'pred_{}.png'.format(i), T1.clip(0,T1.max()*0.7), cmap='gray')
                plt.imsave(visual_dir + 'error_{}.png'.format(i), err_T1,vmax=0.5)

                # pred_list.append(T1)
                # error_list.append(err_T1)
                #
                #
                #
                # plt.imsave(visual_dir + '{}_{}_error.png'.format(self.model_name, str(100 // self.sampling_rate)),
                #            np_visual_stack_error)
                # plt.imsave(visual_dir + '{}_{}_pred.png'.format(self.model_name, str(100 // self.sampling_rate)),
                #            np_visual_stack, cmap='gray')

            # pred_list.append(im_A_[0,0].cpu().numpy())
            # error_list.append(mask_[0,0].cpu().numpy())
            #
            # np_visual_stack = np.hstack(pred_list)
            # np_visual_stack_error = np.hstack(error_list)
            #
            # visual_dir = 'visual/'
            # if not os.path.isdir(visual_dir):
            #     os.makedirs(visual_dir)
            #
            # plt.imsave(visual_dir+'{}_{}_error.png'.format(self.model_name, str( 100 // self.sampling_rate)), np_visual_stack_error)
            # plt.imsave(visual_dir+'{}_{}_pred.png'.format(self.model_name, str( 100 // self.sampling_rate)), np_visual_stack,cmap='gray')


            # skimage.io.imsave('{}_{}.png'.format(self.mask_name, sampling_rate), np_visual_stack)

            # x5
            # d5c5_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1
            # pdnet_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_lpd10-nonzero_init-compound
            # pdnet_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_pdnet-my3_residual_1_5xbuffer_compound-full
            # pdnet_ocmr_cartes_20_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_unet-pdnet-my3_residual_1_5xbuffer_compound-full

            # x10
            # d5c5_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_d5c5_compound-full
            # pdnet_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_unet-pdnet-my3_residual_1_5xbuffer_compound-full
            # pdnet_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_pdnet-my3_residual_1_5xbuffer_compound-full
            # pdnet_ocmr_cartes_10_lr_-3.0_end_factor_1.0_beta_0_norm_type_0_bs_1_loss_compound_aug_1_pdnet_5xbuffer_compound-full