# @author       : Bingyu Xin   
# @Institute    : CS@Rutgers

from model.ISTANet_plus import ISTANetplus
from model.DCCNN import DCCNN
from model.HQSNet import HQSNet
from model.LPDNet import LPDNet
import torch

# net = ISTANetplus()  #DCCNN()#DCCNN()#LPDNet() #HQSNet()#
# im_un = torch.zeros((1, 2, 64, 64))
# k_un = torch.zeros((1, 2, 64, 64))
# mask = torch.zeros((1, 2, 64, 64))
# with torch.no_grad():
#     y = net(im_un, k_un, mask)
#     print(net)
#     # print(loss[1].shape)
# print('Total # of params: %.5fM' % (sum(p.numel() for p in net.parameters()) / (1024.0 * 1024)))

#iter=8;conv=6;
## dc-cnn: Total # of params: 1.14504M; 0.15184M
# ISTANetplus: Total # of params: 1.14259M; 0.36257M
# LPDNet: Total # of params: 2.45718M; 0.30354M
# HQSNet: Total # of params: 1.22420M; 0.14902M

from torchvision.models import resnet50
# from thop import profile
from fvcore.nn import FlopCountAnalysis

model_name = 'ista-net-plus' #'hqs-net-unet'#'lpd-net'#
if model_name == 'dc-cnn':
    net = DCCNN(n_iter=8)
elif model_name == 'ista-net-plus':
    net = ISTANetplus(n_iter=8)
elif model_name == 'lpd-net':
    net = LPDNet(n_iter=8)
elif model_name == 'hqs-net':
    net = HQSNet(block_type='cnn', buffer_size=5, n_iter=8)
elif model_name == 'hqs-net-unet':
    net = HQSNet(block_type='unet', n_iter=10)

net.cuda()
# net = ISTANetplus().cuda()  #DCCNN()#DCCNN()#LPDNet() #HQSNet()#
im_A_und = torch.zeros((1, 2, 192, 160)).cuda()
k_A_und = torch.zeros((1, 2, 192, 160)).cuda()
mask = torch.zeros((1, 2, 192, 160)).cuda()
flops = FlopCountAnalysis(net, (im_A_und, k_A_und, mask))

# macs, params = profile(net, inputs=(im_A_und, k_A_und, mask))
print('Total # of params: %.5fM' % (sum(p.numel() for p in net.parameters()) / 10.**6))
print('Total # of params: %.5fG' % ((flops.total()) / 10.**9))