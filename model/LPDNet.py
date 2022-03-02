# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import torch
from torch import nn
from model.BasicModule import conv_block


class LPDNet(nn.Module):
    def __init__(self, n_primal=5, n_dual=5, n_iter=8, n_convs=6, n_filters=64, norm='ortho'):
        '''
        LPD-Net modified from paper " Learned primal-dual reconstruction "
        ( https://arxiv.org/abs/1707.06474 ) ( https://github.com/adler-j/learned_primal_dual )
        :param n_primal: buffer size for primal space ( image space )
        :param n_dual: buffer size for dual space ( k-space )
        :param n_iter: num of iterations
        :param n_convs: num of convs in the block
        :param n_filters: num of feature channels in intermediate features
        :param norm: 'ortho' norm for fft
        '''
        super().__init__()
        self.norm = norm
        self.n_primal = n_primal
        self.n_dual = n_dual
        self.n_iter = n_iter
        image_net_block = []
        kspace_net_block = []

        for i in range(self.n_iter):
            image_net_block.append(
                conv_block('prim-net', channel_in=2 * (self.n_primal + 1), n_convs=n_convs, n_filters=n_filters))
        self.primal_net = nn.ModuleList(image_net_block)

        for i in range(self.n_iter):
            kspace_net_block.append(
                conv_block('dual-net', channel_in=2 * (self.n_dual + 2), n_convs=n_convs, n_filters=n_filters))
        self.dual_net = nn.ModuleList(kspace_net_block)

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

    def forward(self, img, k, mask):

        dual_buffer = torch.cat([k] * self.n_dual, 1).to(k.device)
        primal_buffer = torch.cat([img] * self.n_primal, 1).to(k.device)

        for i in range(self.n_iter):  #
            # kspace (dual)
            f_2 = primal_buffer[:, 2:4].clone()
            dual_buffer = dual_buffer + self.dual_net[i](
                torch.cat([dual_buffer, self._forward_operation(f_2, mask), k], 1)
            )
            h_1 = dual_buffer[:, 0:2].clone()
            # image space (primal)
            primal_buffer = primal_buffer + self.primal_net[i](
                torch.cat([primal_buffer, self._backward_operation(h_1, mask)], 1)
            )

        return primal_buffer[:, 0:2]
