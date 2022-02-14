# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers
import torch
from torch import nn
from model.BasicModule import conv_block
from model.BasicModule import UNetRes


class HQSNet(nn.Module):
    def __init__(self, buffer_size=5, n_iter=8, n_convs=6, n_filters=64, block_type='cnn', norm='ortho'):
        '''
        HQS-Net
        :param buffer_size: buffer_size m
        :param n_iter: iterations n
        :param block: block type: 'cnn' or 'unet'
        :param norm: 'ortho' norm for fft
        '''
        super().__init__()
        self.norm = norm
        self.m = buffer_size
        self.n_iter = n_iter
        ## the initialization of mu may influence the final accuracy
        self.mu = nn.Parameter(0.5 * torch.ones((1, 1))) #2
        self.block_type = block_type
        if self.block_type == 'cnn':
            rec_blocks = []
            for i in range(self.n_iter):
                rec_blocks.append(
                    conv_block('hqs-net', channel_in=2 * (self.m+1 ), n_convs=n_convs, n_filters=n_filters)) #self.m +
            self.rec_blocks = nn.ModuleList(rec_blocks)
        elif self.block_type == 'unet':
            self.rec_blocks = UNetRes(in_nc=2 * (self.m + 1), out_nc=2 * self.m, nc=[64, 128, 256, 512], nb=4,
                                      act_mode='R',
                                      downsample_mode="strideconv", upsample_mode="convtranspose")

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

    def forward(self, img, k, mask):

        ## initialize buffer f : the concatenation of m copies of the complex-valued zero-filled images

        f = torch.cat([img] * self.m, 1).to(img.device)

        ## n reconstruction blocks  buff=5_nocat
        # for i in range(self.n_iter):
        #     for j in range(self.m):
        #         f_1 = f[:, j*2:j*2+2].clone()
        #         f[:, j*2:j*2+2] = self.update_opration(f_1, k, mask)
        #     if self.block_type == 'cnn':
        #         # f = f + self.rec_blocks[i](torch.cat([f, updated_f_1], 1))
        #         f = f + self.rec_blocks[i](f)
        #     elif self.block_type == 'unet':
        #         f = f + self.rec_blocks(torch.cat([f, updated_f_1], 1))

        ## n reconstruction blocks
        for i in range(self.n_iter):
            f_1 = f[:, 0:2].clone()
            updated_f_1 = self.update_opration(f_1, k, mask)
            if self.block_type == 'cnn':
                f = f + self.rec_blocks[i](torch.cat([f, updated_f_1], 1))
                # f = updated_f_1 + self.rec_blocks[i](updated_f_1)
            elif self.block_type == 'unet':
                f = f + self.rec_blocks(torch.cat([f, updated_f_1], 1))
        return f[:, 0:2]


if __name__ == '__main__':
    net = HQSNet()  #
    im_un = torch.zeros((1, 2, 64, 64))
    k_un = torch.zeros((1, 2, 64, 64))
    mask = torch.zeros((1, 2, 64, 64))
    with torch.no_grad():
        y = net(im_un, k_un, mask)
    print('Total # of params: %.5fM' % (sum(p.numel() for p in net.parameters()) / (1024.0 * 1024)))
