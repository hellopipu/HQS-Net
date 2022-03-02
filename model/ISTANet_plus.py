# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self, norm='ortho'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        num_filter = 64

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, 2, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(num_filter, num_filter, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(2, num_filter, 3, 3)))

    def _forward_operation(self, img, mask):

        k = torch.fft.fft2(torch.view_as_complex(img.permute(0, 2, 3, 1).contiguous()),
                           norm=self.norm)
        k = torch.view_as_real(k).permute(0, 3, 1, 2).contiguous()
        k = mask * k
        return k

    def _backward_operation(self, k, mask):

        k = mask * k
        img = torch.fft.ifft2(torch.view_as_complex(k.permute(0, 2, 3, 1).contiguous()), norm=self.norm)
        img = torch.view_as_real(img).permute(0, 3, 1, 2).contiguous()
        return img

    def update_opration(self, f_1, k, mask):
        h_1 = k - self._forward_operation(f_1, mask)
        update = f_1 + self.lambda_step * self._backward_operation(h_1, mask)
        return update

    def forward(self, x, k, m):
        # x = x - self.lambda_step * fft_forback(x, mask)
        # x = x + self.lambda_step * PhiTb
        x = self.update_opration(x, k, m)
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_G = F.conv2d(x_backward, self.conv_G, padding=1)

        x_pred = x_input + x_G

        if self.training:
            x = F.conv2d(x_forward, self.conv1_backward, padding=1)
            x = F.relu(x)
            x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
            symloss = x_D_est - x_D
            return x_pred, symloss
        else:
            return x_pred, None


class ISTANetplus(nn.Module):
    def __init__(self, n_iter=8, n_convs=5, n_filters=64, norm='ortho'):
        '''
        ISTANetplus modified from paper " ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image
Compressive Sensing "
        ( https://arxiv.org/pdf/1706.07929.pdf ) ( https://github.com/jianzhangcs/ISTA-Net-PyTorch )
        :param n_iter: num of iterations
        :param n_convs: num of convs in each block
        :param n_filters: num of feature channels in intermediate features
        :param norm: 'ortho' norm for fft
        '''
        super(ISTANetplus, self).__init__()
        channel_in = 2
        rec_blocks = []
        self.norm = norm
        self.n_iter = n_iter
        for i in range(n_iter):
            rec_blocks.append(BasicBlock(norm=self.norm))
        self.rec_blocks = nn.ModuleList(rec_blocks)

    def forward(self, x, k, m):
        layers_sym = []  # for computing symmetric loss
        for i in range(self.n_iter):
            x, layer_sym = self.rec_blocks[i](x, k, m)
            layers_sym.append(layer_sym)
        if self.training:
            return x, layers_sym
        else:
            return x
