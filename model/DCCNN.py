# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import torch
import torch.nn as nn
from model.BasicModule import conv_block


class DCCNN(nn.Module):
    def __init__(self, n_iter=5, n_convs=5, n_filters=64, norm='ortho'):
        '''
        DC-CNN modified from paper " A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction "
        ( https://arxiv.org/pdf/1704.02422.pdf ) ( https://github.com/js3611/Deep-MRI-Reconstruction )
        :param n_iter: num of iterations
        :param n_convs: num of convs in each block
        :param n_filters: num of feature channels in intermediate features
        :param norm: 'ortho' norm for fft
        '''
        super(DCCNN, self).__init__()
        channel_in = 2
        rec_blocks = []
        self.norm = norm
        self.n_iter = n_iter
        for i in range(n_iter):
            rec_blocks.append(conv_block('dc-cnn', channel_in, n_filters=n_filters, n_convs=n_convs))

        self.rec_blocks = nn.ModuleList(rec_blocks)

    def dc_operation(self, x_rec, k_un, mask):
        x_rec = x_rec.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        k_un = k_un.permute(0, 2, 3, 1)
        k_rec = torch.fft.fft2(torch.view_as_complex(x_rec.contiguous()), norm=self.norm)

        k_rec = torch.view_as_real(k_rec)
        k_out = k_rec + (k_un - k_rec) * mask

        k_out = torch.view_as_complex(k_out)
        x_out = torch.view_as_real(torch.fft.ifft2(k_out, norm=self.norm))
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out

    def forward(self, x, k, m):
        for i in range(self.n_iter):
            x_cnn = self.rec_blocks[i](x)
            x = x + x_cnn
            x = self.dc_operation(x, k, m)
        return x


if __name__ == '__main__':
    net = DCCNN()  #
    im_un = torch.zeros((1, 2, 64, 64))
    k_un = torch.zeros((1, 2, 64, 64))
    mask = torch.zeros((1, 2, 64, 64))
    with torch.no_grad():
        y = net(im_un, k_un, mask)
    print('Total # of params: %.5fM' % (sum(p.numel() for p in net.parameters()) / (1024.0 * 1024)))
