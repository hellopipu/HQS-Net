# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import os
from os.path import join as join
import glob
import random

import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomApply, RandomRotation, ToTensor, RandomResizedCrop, \
    Compose, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective
from utils import undersample, cartesian_mask
from os.path import splitext
from os import listdir, path
import h5py
import random
import xml.etree.ElementTree as etree
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import h5py
import numpy as np
import torch
from tqdm import tqdm


class MyData(Dataset):
    def __init__(self, imageDir, maskDir=None, sample_rate=8, img_size=256, is_training='train', dataset='knee_100',gen2=False,norm=False):
        super().__init__()
        self.dataset = dataset
        self.norm=norm # norm type of data
        self.img_size = img_size  ## used in transform
        self.gen2 = gen2  # for refinegan to generate 2 imgs in getitem

        ### settings for calgary dataset
        self.sample_rate = sample_rate
        self.is_training = is_training
        if self.dataset == 'knee_100':
            images = glob.glob(join(imageDir, '*'))
            self.images = sorted(images)
            ## get mask, use only one mask for knee_100 dataset in train and val
            masks = glob.glob(join(maskDir, '*'))
            mask = skimage.io.imread(masks[0])
            self.mask = torch.from_numpy(mask / 255.)
        elif self.dataset == 'calgary':
            self.images = np.load(imageDir)
            # print('debug: ',self.images.shape)
            # self.mask = torch.from_numpy(cartesian_mask((1,256,256), acc, centred= False, sample_random=False)[0])
            ## get mask, use only one mask for knee_100 dataset in train and val
            mask = cartesian_mask(self.img_size, acc=100 / self.sample_rate, centred=False, sample_random=False)
            self.mask = torch.from_numpy(mask)
        elif self.dataset == 'ocmr':
            self.images = np.load(imageDir)
            # self.mask = torch.from_numpy(cartesian_mask((1,256,256), acc, centred= False, sample_random=False)[0])
            ## get mask, use only one mask for knee_100 dataset in train and val
            mask = cartesian_mask(self.img_size, acc=100/self.sample_rate, centred = False, sample_random=False)
            self.mask = torch.from_numpy(mask)
        self.len = len(self.images)
        self.custom_transform = [
            ToTensor(),
            RandomApply(
                torch.nn.ModuleList([RandomResizedCrop(self.img_size, scale=(0.9, 1.0), ratio=(0.9, 1.1))]),
                p=0.3),
            RandomApply(torch.nn.ModuleList([RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1),
                                                          shear=(-5, 5, -5, 5),
                                                          interpolation=transforms.InterpolationMode.BILINEAR)]),
                        p=0.3),
            # RandomHorizontalFlip(p=0.3),
            # RandomVerticalFlip(p=0.3),
            # RandomPerspective(0.05, 0.3),
        ]

    def transform(self, img_A, norm, use_transform):
        '''

        :param img_A: numpy array, (2,H,W)
        :param norm:
        :param use_transform:
        :return: torch tensor, complex, (H,W)
        '''
        if self.dataset == 'knee_100':
            img_A = img_A / 255.
        elif self.dataset == 'calgary' or self.dataset == 'ocmr':
            # img_A = img_A.clip(0.,img_A.max()*0.8)
            img_A = img_A/img_A.max()

        if use_transform:
            img_A = img_A.transpose(1,2,0)
            for t in self.custom_transform:
                img_A = t(img_A)
        else:
            img_A = torch.from_numpy(img_A)

        img_A = img_A[0] + 1j * img_A[1]
        if norm:
            # complex value normalize to [-1-j,1+j],
            # so for the 2 channel real representation, pixel range is [-1.,1.]
            img_A = (img_A - (0.5 + 0.5j)) * 2.0

        return img_A

    def get_sample(self,index, norm, use_transform = False):
        if self.dataset =='knee_100':
            image_A = skimage.io.imread(self.images[index])
        elif  self.dataset == 'calgary' or self.dataset == 'ocmr':
            image_A = self.images[index]
            # image_A = np.fft.ifft2(k_A)
            # image_A = np.stack([np.real(image_A), np.imag(image_A)], axis=0).astype(np.float32)

        ########################### image preprocessing ###########################
        # random transform to image A and B , and norm
        image_A = self.transform(image_A,norm,use_transform)
        # generate zero-filled image x_und, k_und, k
        image_A_und, k_A_und, k_A = undersample(image_A, self.mask)

        ########################## complex to 2 channel ##########################
        im_A = torch.view_as_real(image_A).permute(2, 0, 1).contiguous()
        im_A_und = torch.view_as_real(image_A_und).permute(2, 0, 1).contiguous()
        k_A_und = torch.view_as_real(k_A_und).permute(2, 0, 1).contiguous()

        return im_A,im_A_und,k_A_und

    def __getitem__(self, i):
        if self.is_training == 'train':

            im_A, im_A_und, k_A_und = self.get_sample(i,norm=self.norm, use_transform=True)
            mask = torch.view_as_real(self.mask * (1. + 1.j)).permute(2, 0, 1).contiguous()
            data_dict = {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und,'mask_A': mask}
            if self.gen2:
                index_B = random.randint(0, self.len - 1)
                im_B, im_B_und, k_B_und = self.get_sample(index_B,norm=self.norm, use_transform=True)
                B_dict = {'im_B': im_B, 'im_B_und': im_B_und, 'k_B_und': k_B_und}
                data_dict.update(B_dict)
            return data_dict

        else:

            im_A, im_A_und, k_A_und = self.get_sample(i,norm=self.norm, use_transform=False)
            mask = torch.view_as_real(self.mask * (1. + 1.j)).permute(2, 0, 1).contiguous()

            return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und, 'mask_A': mask}

    def __len__(self):
        return self.len






