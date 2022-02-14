# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import torch
import torch.nn as nn
from model.BasicModule import conv_block


class DCCNN(nn.Module):
    def __init__(self, n_iter=8, n_convs=6, n_filters=64, norm='ortho'):
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
        self.mu = nn.Parameter(torch.Tensor([0.5]))
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
        # noiseless
        k_out = k_rec + (k_un - k_rec) * mask

        k_out = torch.view_as_complex(k_out)
        x_out = torch.view_as_real(torch.fft.ifft2(k_out, norm=self.norm))
        x_out = x_out.permute(0, 3, 1, 2)
        return x_out
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
        update = f_1 + self.mu * self._backward_operation(h_1, mask)
        return update

    def forward(self, x, k, m):
        for i in range(self.n_iter):
            # x = self.update_opration(x, k, m)
            x_cnn = self.rec_blocks[i](x)
            x = x + x_cnn
            x = self.update_opration(x, k, m)
        return x


if __name__ == '__main__':
    net = DCCNN()  #
    im_un = torch.zeros((1, 2, 64, 64))
    k_un = torch.zeros((1, 2, 64, 64))
    mask = torch.zeros((1, 2, 64, 64))
    with torch.no_grad():
        y = net(im_un, k_un, mask)
    print('Total # of params: %.5fM' % (sum(p.numel() for p in net.parameters()) / (1024.0 * 1024)))
