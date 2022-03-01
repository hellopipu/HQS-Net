# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import os
from os.path import join
import time
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from skimage.metrics import structural_similarity as cal_ssim
from skimage.metrics import peak_signal_noise_ratio as cal_psnr
from skimage.metrics import normalized_root_mse as cal_nrmse

from loss import CompoundLoss
from utils import output2complex
from read_data import MyData
from model.DCCNN import DCCNN
from model.LPDNet import LPDNet
from model.HQSNet import HQSNet
from model.ISTANet_plus import ISTANetplus
import numpy as np

class Solver():
    def __init__(self, args):
        torch.autograd.set_detect_anomaly(True)
        self.args = args
        ################  experiment settings  ################
        self.model_name = self.args.model

        self.acc = self.args.acc  # for ocmr dataset only
        self.imageDir_train = self.args.train_path  # train path
        self.imageDir_val = self.args.val_path  # val path while training
        self.imageDir_test = self.args.test_path  # test path
        self.num_epoch = self.args.num_epoch
        self.batch_size = self.args.batch_size  # batch size
        self.val_on_epochs = self.args.val_on_epochs  # val on each val_on_epochs epochs;

        ## settings for optimizer
        self.lr = self.args.lr
        ## settings for data preprocessing
        self.img_size = (192, 160)
        self.saveDir = 'weight'  # model save path while training
        if not os.path.isdir(self.saveDir):
            os.makedirs(self.saveDir)

        self.task_name = self.model_name + '_acc_' + str(self.acc) + '_bs_' + str(self.batch_size) \
                         + '_lr_' + str(self.lr) # + 'bf_5_nocat' #first_dc'#'_iter_10'#'_bf=1' #+  _nocat '_bf=1'#
        print('task_name: ', self.task_name)
        self.model_path = 'weight/' + self.task_name + '_best.pth'  # model load path for test and visualization

        ############################################ Specify network ############################################
        if self.model_name == 'dc-cnn':
            self.net = DCCNN(n_iter=8)
        elif self.model_name == 'ista-net-plus':
            self.net = ISTANetplus(n_iter=8)
        elif self.model_name == 'lpd-net':
            self.net = LPDNet(n_iter=8)
        elif self.model_name == 'hqs-net':
            self.net = HQSNet(block_type='cnn',buffer_size=5, n_iter=8)
        elif self.model_name == 'hqs-net-unet':
            self.net = HQSNet(block_type='unet',buffer_size=5, n_iter=10)
        else:
            assert "wrong model name !"
        print('Total # of model params: %.5fM' % (sum(p.numel() for p in self.net.parameters()) / 10.**6))
        self.net.cuda()

    def train(self):

        ############################################ Specify loss ############################################
        ## Notice:
        ## 0. generally, ms-ssim is slightly better than ssim
        ## 1. we train all models with ms-ssim + l1 loss, except hqs-net-unet
        ## 2. we train the hqs-net-unet model with ssim + l1 loss, the reason is that, we found when using ms-ssim loss,
        ## the gradient of ms-ssim may be nan. This bug exists in both pytorch and tensoflow implementation of ms-ssim loss.
        ## see https://github.com/tensorflow/tensorflow/issues/50400, https://github.com/VainF/pytorch-msssim/issues/12
        # if self.model_name == 'hqs-net-unet':
        #     self.criterion = CompoundLoss('ssim')
        # else:
        self.criterion = CompoundLoss('ms-ssim')

        ############################################ Specify optimizer ########################################

        self.optimizer_G = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-3, weight_decay=1e-10)

        ############################################ load data ############################################

        dataset_train = MyData(self.imageDir_train, self.acc, self.img_size, is_training='train')
        dataset_val = MyData(self.imageDir_val, self.acc, self.img_size, is_training='val')

        num_workers = 4
        use_pin_memory = True
        loader_train = Data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=num_workers, pin_memory=use_pin_memory)
        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=num_workers, pin_memory=use_pin_memory)
        self.slices_val = len(dataset_val)
        print("slices of 2d train data: ", len(dataset_train))
        print("slices of 2d validation data: ", len(dataset_val))

        ############################################ setting for tensorboard ###################################
        self.writer = SummaryWriter('log/' + self.task_name)

        ############################################ start to run epochs #######################################

        start_epoch = 0
        best_val_psnr = 0
        if 0:
            best_name = self.task_name + '_best.pth'
            checkpoint = torch.load(join(self.saveDir, best_name))
            self.net.load_state_dict(checkpoint['net'])
            start_epoch = checkpoint['epoch']+1
            best_val_psnr = checkpoint['val_psnr']
            print('load pretrained model---, start epoch at, ',start_epoch, ', star_psnr_val is: ',best_val_psnr)
        for epoch in range(start_epoch, self.num_epoch):
            ####################### 1. training #######################

            loss_g = self._train_cnn(loader_train)
            ####################### 2. validate #######################
            if epoch == start_epoch:
                base_psnr, base_ssim = self._validate_base(loader_val)
            if epoch % self.val_on_epochs == 0:
                val_psnr, val_ssim = self._validate(loader_val)
                ########################## 3. print and tensorboard ########################
                print("Epoch {}/{}".format(epoch + 1, self.num_epoch))
                print(" base PSNR:\t\t{:.6f}".format(base_psnr))
                print(" test PSNR:\t\t{:.6f}".format(val_psnr))
                print(" base SSIM:\t\t{:.6f}".format(base_ssim))
                print(" test SSIM:\t\t{:.6f}".format(val_ssim))
                ## write to tensorboard
                self.writer.add_scalar("loss/train_loss", loss_g, epoch)
                self.writer.add_scalar("metric/base_psnr", base_psnr, epoch)
                self.writer.add_scalar("metric/val_psnr", val_psnr, epoch)
                self.writer.add_scalar("metric/base_ssim", base_ssim, epoch)
                self.writer.add_scalar("metric/val_ssim", val_ssim, epoch)
                ## save the best model according to validation psnr
                if best_val_psnr < val_psnr:
                    best_val_psnr = val_psnr
                    best_name = self.task_name + '_best.pth'
                    state = {'net': self.net.state_dict(), 'epoch': epoch, 'val_psnr': val_psnr, 'val_ssim': val_ssim}
                    torch.save(state, join(self.saveDir, best_name))
        self.writer.close()

    def test(self):

        ############################################ load data ################################

        dataset_val = MyData(self.imageDir_test, self.acc, self.img_size, is_training='test')

        loader_val = Data.DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                     num_workers=4, pin_memory=True)
        len_data = len(dataset_val)
        print("slices of 2d test data: ", len_data)
        checkpoint = torch.load(self.model_path)

        print("best epoch at : {}, val_psnr: {:.6f}, val_ssim: {:.6f}" \
              .format(checkpoint['epoch'], checkpoint['val_psnr'], checkpoint['val_ssim']))

        self.net.load_state_dict(checkpoint['net'])
        self.net.cuda()
        self.net.eval()

        base_psnr = []
        test_psnr = []
        base_ssim = []
        test_ssim = []
        base_nrmse = []
        test_nrmse = []
        with torch.no_grad():
            time_0 = time.time()
            for data_dict in tqdm(loader_val):
                im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict['im_A_und'].float().cuda(), \
                                                data_dict['k_A_und'].float().cuda(), \
                                                data_dict['mask_A'].float().cuda()

                if self.model_name == 'ista-net-plus':
                    T1, loss_layers_sym = self.net(im_A_und, k_A_und, mask)
                else:
                    T1 = self.net(im_A_und, k_A_und, mask)
                ############## convert model ouput to complex value in original range

                T1 = output2complex(T1)
                im_A = output2complex(im_A)
                im_A_und = output2complex(im_A_und)

                ########################### calulate metrics ###################################
                for T1_i, im_A_i, im_A_und_i in zip(T1.cpu().numpy(), im_A.cpu().numpy(), im_A_und.cpu().numpy()):
                    ## for skimage.metrics, input is (im_true,im_pred)
                    base_nrmse.append(cal_nrmse(im_A_i, im_A_und_i))
                    test_nrmse.append(cal_nrmse(im_A_i, T1_i))
                    base_ssim.append(cal_ssim(im_A_i, im_A_und_i, data_range=im_A_i.max()))
                    test_ssim.append(cal_ssim(im_A_i, T1_i, data_range=im_A_i.max()))
                    base_psnr.append(cal_psnr(im_A_i, im_A_und_i, data_range=im_A_i.max()))
                    test_psnr.append(cal_psnr(im_A_i, T1_i, data_range=im_A_i.max()))

            time_1 = time.time()
            ## comment metric calculation code for more precise inference speed
            print('inference speed: {:.5f} ms/slice'.format(1000 * (time_1 - time_0) / len_data))

        print(" base PSNR:\t\t{:.6f}, std: {:.6f}".format(np.mean(base_psnr),np.std(base_psnr)))
        print(" test PSNR:\t\t{:.6f}, std: {:.6f}".format(np.mean(test_psnr),np.std(test_psnr)))
        print(" base SSIM:\t\t{:.6f}, std: {:.6f}".format(np.mean(base_ssim),np.std(base_ssim)))
        print(" test SSIM:\t\t{:.6f}, std: {:.6f}".format(np.mean(test_ssim),np.std(test_ssim)))
        print(" base NRMSE:\t\t{:.6f}, std: {:.6f}".format(np.mean(base_nrmse),np.std(base_nrmse)))
        print(" test NRMSE:\t\t{:.6f}, std: {:.6f}".format(np.mean(test_nrmse),np.std(test_nrmse)))

    def _train_cnn(self, loader_train):
        self.net.train()
        for data_dict in tqdm(loader_train):
            im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                'im_A_und'].float().cuda(), data_dict['k_A_und'].float().cuda(), data_dict['mask_A'].float().cuda()
            if self.model_name == 'ista-net-plus':
                T1,loss_layers_sym = self.net(im_A_und, k_A_und, mask)
            else:
                T1 = self.net(im_A_und, k_A_und, mask)

            T1 = output2complex(T1)
            im_A = output2complex(im_A)
            ############################################# 1.2 update generator #############################################

            loss_g = self.criterion(T1, im_A, data_range=im_A.max())
            if self.model_name == 'ista-net-plus':
                loss_constraint = torch.mean(torch.pow(loss_layers_sym[0], 2))
                for k in range(len(loss_layers_sym)-1):
                    loss_constraint += torch.mean(torch.pow(loss_layers_sym[k + 1], 2))
                loss_g = loss_g + 0.01 * loss_constraint

            self.optimizer_G.zero_grad()
            loss_g.backward()
            self.optimizer_G.step()

        return loss_g

    def _validate_base(self, loader_val):

        base_psnr = 0
        base_ssim = 0

        for data_dict in loader_val:
            im_A, im_A_und, = data_dict['im_A'].float().cuda(), data_dict['im_A_und'].float().cuda()
            ############## convert model ouput to complex value in original range
            im_A = output2complex(im_A)
            im_A_und = output2complex(im_A_und)
            ########################### cal metrics ###################################
            for im_A_i, im_A_und_i in zip(im_A.cpu().numpy(),
                                          im_A_und.cpu().numpy()):
                ## for skimage.metrics, input is (im_true,im_pred)
                base_ssim += cal_ssim(im_A_i, im_A_und_i, data_range=im_A_i.max())
                base_psnr += cal_psnr(im_A_i, im_A_und_i, data_range=im_A_i.max())
        base_psnr /= self.slices_val
        base_ssim /= self.slices_val
        return base_psnr, base_ssim

    def _validate(self, loader_val):

        test_psnr = 0
        test_ssim = 0

        self.net.eval()
        with torch.no_grad():
            for data_dict in tqdm(loader_val):

                im_A, im_A_und, k_A_und, mask = data_dict['im_A'].float().cuda(), data_dict[
                    'im_A_und'].float().cuda(), data_dict['k_A_und'].float().cuda(), data_dict[
                                                    'mask_A'].float().cuda()
                if self.model_name == 'ista-net-plus':
                    T1,_ = self.net(im_A_und, k_A_und, mask)
                else:
                    T1 = self.net(im_A_und, k_A_und, mask)
                ############## convert model ouput to complex value in original range
                T1 = output2complex(T1)
                im_A = output2complex(im_A)

                ########################### cal metrics ###################################
                for T1_i, im_A_i in zip(T1.cpu().numpy(), im_A.cpu().numpy()):
                    ## for skimage.metrics, input is (im_true,im_pred)
                    test_ssim += cal_ssim(im_A_i, T1_i, data_range=im_A_i.max())
                    test_psnr += cal_psnr(im_A_i, T1_i, data_range=im_A_i.max())

        test_psnr /= self.slices_val
        test_ssim /= self.slices_val
        return test_psnr, test_ssim
