# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import torch
import torch.nn as nn
from model.BasicModule import conv_block
import torch.functional as F
from torch.nn import init


# Define ISTA-Net-plus Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, fft_forback, PhiTb, mask):
        x = x - self.lambda_step * fft_forback(x, mask)
        x = x + self.lambda_step * PhiTb
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

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


class ISTANet(nn.Module):
    def __init__(self, n_iter=5, n_convs=5, n_filters=64, norm='ortho'):
        '''
        DC-CNN modified from paper " A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction "
        ( https://arxiv.org/pdf/1704.02422.pdf ) ( https://github.com/js3611/Deep-MRI-Reconstruction )
        :param n_iter: num of iterations
        :param n_convs: num of convs in each block
        :param n_filters: num of feature channels in intermediate features
        :param norm: 'ortho' norm for fft
        '''
        super(ISTANet, self).__init__()
        channel_in = 2
        rec_blocks_forward = []
        rec_blocks_backward = []
        self.norm = norm
        self.n_iter = n_iter
        self.mu = nn.Parameter(0.5 * torch.ones(self.n_iter))
        self.soft_thr = nn.Parameter(0.01 * torch.ones(self.n_iter))
        for i in range(n_iter):
            rec_blocks_forward.append(conv_block('ista-net', channel_in, n_filters=n_filters, n_convs=n_convs))
            rec_blocks_backward.append(conv_block('dc-cnn', channel_in, n_filters=n_filters, n_convs=n_convs))

        self.rec_blocks_forward = nn.ModuleList(rec_blocks_forward)
        self.rec_blocks_backward = nn.ModuleList(rec_blocks_backward)

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

    def update_opration(self, f_1, k, mask, i):
        h_1 = k - self._forward_operation(f_1, mask)
        update = f_1 + self.mu[i] * self._backward_operation(h_1, mask)
        return update

    def forward(self, x, k, m):
        for i in range(self.n_iter):
            x_input = self.update_opration(x, k, mask, i)
            x_forward = self.rec_blocks_forward[i](x_input)
            x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr[i]))
            x = self.rec_blocks_backward[i](x)
            x = x + x_input
            symloss =
        return x


if __name__ == '__main__':
    net = DCCNN()  #
    im_un = torch.zeros((1, 2, 64, 64))
    k_un = torch.zeros((1, 2, 64, 64))
    mask = torch.zeros((1, 2, 64, 64))
    with torch.no_grad():
        y = net(im_un, k_un, mask)
    print('Total # of params: %.5fM' % (sum(p.numel() for p in net.parameters()) / (1024.0 * 1024)))
