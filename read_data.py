# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import RandomApply, RandomRotation, ToTensor, RandomResizedCrop, \
    Compose, RandomAffine, RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective
from utils import undersample, cartesian_mask


class MyData(Dataset):
    def __init__(self, imageDir, acc=5, img_size=256, is_training='train'):
        super().__init__()

        self.img_size = img_size  ## used in transform
        self.is_training = is_training
        self.acc = acc

        self.images = np.load(imageDir)

        ## generate a fixed mask for validating and testing
        if self.is_training != 'train':
            mask = cartesian_mask(self.img_size, acc=self.acc, centred=False, sample_random=False)
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

    def transform(self, img_A, use_transform):
        '''

        :param img_A: numpy array, (2,H,W)
        :param use_transform:
        :return: torch tensor, complex, (H,W)
        '''
        ## normalize to [0,1]
        img_A = img_A / img_A.max()

        if use_transform:
            img_A = img_A.transpose(1, 2, 0)
            for t in self.custom_transform:
                img_A = t(img_A)
        else:
            img_A = torch.from_numpy(img_A)

        img_A = img_A[0] + 1j * img_A[1]

        return img_A

    def get_sample(self, index, mask, use_transform=False):

        image_A = self.images[index]

        ########################### image preprocessing ###########################
        # random transform
        image_A = self.transform(image_A, use_transform)
        # generate zero-filled image x_und, k_und, k
        image_A_und, k_A_und, k_A = undersample(image_A, mask)

        ########################## complex to 2 channel ##########################
        im_A = torch.view_as_real(image_A).permute(2, 0, 1).contiguous()
        im_A_und = torch.view_as_real(image_A_und).permute(2, 0, 1).contiguous()
        k_A_und = torch.view_as_real(k_A_und).permute(2, 0, 1).contiguous()

        return im_A, im_A_und, k_A_und

    def __getitem__(self, i):
        if self.is_training == 'train':
            ## generate random masks for training
            mask = cartesian_mask(self.img_size, acc=self.acc, centred=False, sample_random=True)
            mask = torch.from_numpy(mask)
            ## generate training samples with augmentation
            im_A, im_A_und, k_A_und = self.get_sample(i, mask, use_transform=True)
            mask = torch.view_as_real(mask * (1. + 1.j)).permute(2, 0, 1).contiguous()
        else:
            im_A, im_A_und, k_A_und = self.get_sample(i, self.mask, use_transform=False)
            mask = torch.view_as_real(self.mask * (1. + 1.j)).permute(2, 0, 1).contiguous()

        return {'im_A': im_A, 'im_A_und': im_A_und, 'k_A_und': k_A_und, 'mask_A': mask}

    def __len__(self):
        return self.len
