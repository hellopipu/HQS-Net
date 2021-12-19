# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import RF
from pytorch_msssim import MS_SSIM
from utils import output2complex
def total_variant(images):
    '''
    :param images:  [B,C,W,H]
    :return: total_variant
    '''
    pixel_dif1 = images[:, :, 1:, :] - images[:, :, :-1, :]
    pixel_dif2 = images[:, :, :, 1:] - images[:, :, :, :-1]

    tot_var = torch.abs(pixel_dif1).sum([1, 2, 3]) + torch.abs(pixel_dif2).sum([1, 2, 3])

    return tot_var



def build_loss(dis_real, dis_fake):
    '''
    calculate WGAN loss
    '''
    d_loss = torch.mean(dis_fake - dis_real)
    g_loss = -torch.mean(dis_fake)
    return g_loss, d_loss


def cal_loss(S01, S01_k_un, S02, S02_k_un, mask, T1, T2, S1_dis_real, S1_dis_fake,
             T1_dis_fake, S2_dis_real, S2_dis_fake,
             T2_dis_fake, cal_G=True):
    '''
    TODO: input arguments are too much, and some calculation is redundant
    '''
    comp_loss = CompoundLoss() ########################

    G_loss_Aa, D_loss_Aa = build_loss(S1_dis_real, T1_dis_fake)
    G_loss_Bb, D_loss_Bb = build_loss(S2_dis_real, T2_dis_fake)

    G_loss_Ab, D_loss_Ab = build_loss(S1_dis_real, T2_dis_fake)
    G_loss_Ba, D_loss_Ba = build_loss(S2_dis_real, T1_dis_fake)


    if cal_G:

        recon_img_Aa = comp_loss(T1,S01) #torch.mean(torch.abs((S01) - (T1)))
        recon_img_Bb = comp_loss(T2,S02) #torch.mean(torch.abs((S02) - (T2)))
        smoothness_Aa = torch.mean(total_variant(T1))
        smoothness_Bb = torch.mean(total_variant(T2))

        ALPHA = 1e+1
        GAMMA = 1e-0
        DELTA = 1e-4
        RATES = torch.count_nonzero(torch.ones_like(mask)) / 2. / torch.count_nonzero(mask)
        GAMMA = RATES

        g_loss = \
            (G_loss_Aa + G_loss_Bb + G_loss_Ab + G_loss_Ba) + \
            (recon_img_Aa + recon_img_Bb) * 1.00 * ALPHA * RATES + \
            ( smoothness_Aa + smoothness_Bb) * DELTA  # for freq loss, when d5c5 backbone, factor is 1e+3; origin is 1. although
        return g_loss, [G_loss_Aa, recon_img_Aa, smoothness_Aa]
    else:

        d_loss = \
            D_loss_Aa + D_loss_Bb + D_loss_Ab + D_loss_Ba

        return d_loss, [ D_loss_Aa, D_loss_Ab]

# def cal_loss(S01, S01_k_un, S02, S02_k_un, mask, Sp1, S1, Tp1, T1, Sp2, S2, Tp2, T2, S1_dis_real, S1_dis_fake,
#              T1_dis_fake, S2_dis_real, S2_dis_fake,
#              T2_dis_fake, cal_G=True):
#     '''
#     TODO: input arguments are too much, and some calculation is redundant
#     '''
#     # comp_loss = CompoundLoss() ########################
#     G_loss_Aa, D_loss_Aa = build_loss(S1_dis_real, T1_dis_fake)
#
#     if cal_G:
#         recon_frq_Aa = torch.mean(torch.abs(S01_k_un - RF(Tp1, mask)))
#         recon_img_Aa = torch.mean(torch.abs((S01) - (T1))) #comp_loss(output2complex(S02),output2complex(T2)) #
#         smoothness_Aa = torch.mean(total_variant(T1))
#
#         ALPHA = 1e+1
#         GAMMA = 1e-0
#         DELTA = 1e-4
#         RATES = torch.count_nonzero(torch.ones_like(mask)) / 2. / torch.count_nonzero(mask)
#         GAMMA = RATES
#
#         g_loss = \
#             (G_loss_Aa)+ \
#             (recon_img_Aa) * 1e+2 * ALPHA * RATES \
#             + smoothness_Aa * DELTA + recon_frq_Aa  * 1.00 * GAMMA * RATES
#
#         return g_loss, [G_loss_Aa, recon_img_Aa]
#     else:
#         d_loss =  D_loss_Aa
#         return d_loss, [ D_loss_Aa]


class SSIMLoss(nn.Module):
    """
    SSIM loss module.

    From: https://github.com/facebookresearch/fastMRI/blob/master/fastmri/losses.py
    """

    def __init__(self, win_size=7, k1=0.01, k2=0.03):
        """
        Args:
            win_size (int, default=7): Window size for SSIM calculation.
            k1 (float, default=0.1): k1 parameter for SSIM calculation.
            k2 (float, default=0.03): k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y):
        '''

        :param X:  pred
        :param Y:  gt
        :return:
        '''
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        # print(Y.device)
        data_range =  torch.Tensor([Y.max()]).to(Y.device)
        # print(data_range.shape,data_range)
        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class CompoundLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.l1loss = nn.L1Loss()
        # self.ssimloss = SSIMLoss().cuda()
        self.msssim = MS_SSIM(win_size=7, data_range=1., size_average=True, channel=1,K=(0.01, 0.03))
        self.alpha = 0.84

    def forward(self, pred, target):
        # pred_abs = pred_abs.unsqueeze(1)
        # target_abs = target_abs.unsqueeze(1)
        # data_range = target_abs.max() #
        l1_loss = self.l1loss(pred, target)
        ssim_loss = 1 - self.msssim(pred.unsqueeze(1), target.unsqueeze(1))#self.ssimloss(pred.abs(), target.abs())
        return (1-self.alpha)*l1_loss + self.alpha * ssim_loss

def gradient_penalty( y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda()
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)


def bayeGen_loss(out_mean, out_1alpha, out_beta, target):
    alpha_eps, beta_eps = 1e-5, 1e-1
    out_1alpha = out_1alpha + alpha_eps
    out_beta = out_beta +beta_eps
    factor = out_1alpha
    resi = torch.abs(out_mean - target)
    #     resi = (torch.log((resi*factor).clamp(min=1e-4, max=5))*out_beta).clamp(min=-1e-4, max=5)
    resi = (resi * factor * out_beta).clamp(min=1e-6, max=50)
    log_1alpha = torch.log(out_1alpha)
    log_beta = torch.log(out_beta)
    lgamma_beta = torch.lgamma(torch.pow(out_beta, -1))

    if torch.sum(log_1alpha != log_1alpha) > 0:
        print('log_1alpha has nan')
        print(lgamma_beta.min(), lgamma_beta.max(), log_beta.min(), log_beta.max())
    if torch.sum(lgamma_beta != lgamma_beta) > 0:
        print('lgamma_beta has nan')
    if torch.sum(log_beta != log_beta) > 0:
        print('log_beta has nan')

    l = resi - log_1alpha + lgamma_beta - log_beta
    l = torch.mean(l)
    return l