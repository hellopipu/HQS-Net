# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

import torch
from fvcore.nn import FlopCountAnalysis

from model.ISTANet_plus import ISTANetplus
from model.DCCNN import DCCNN
from model.HQSNet import HQSNet
from model.LPDNet import LPDNet

net1 = DCCNN(n_iter=8)
net2 = ISTANetplus(n_iter=8)
net3 = LPDNet(n_iter=8)
net4 = HQSNet(block_type='cnn', buffer_size=5, n_iter=8)
net5 = HQSNet(block_type='unet', n_iter=10)
net = [net1, net2, net3, net4, net5]
model_name = ['dc-cnn', 'ista-net-plus', 'lpd-net', 'hqs-net', 'hqs-net-unet']

im_A_und = torch.randn((1, 2, 192, 160)).cuda()
k_A_und = torch.randn((1, 2, 192, 160)).cuda()
mask = torch.randn((1, 2, 192, 160)).cuda()
for i in range(len(net)):
    flops = FlopCountAnalysis(net[i].cuda().eval(), (im_A_und, k_A_und, mask))
    ## ignore the information for unspported operation when calculating flops
    flops._enable_warn_unsupported_ops = False
    print('--Information of ' + model_name[i] + ': ')
    print('     Total # of params: %.5fM' % (sum(p.numel() for p in net[i].parameters()) / 10. ** 6))
    print('     Total # of params: %.5fG' % ((flops.total()) / 10. ** 9))
