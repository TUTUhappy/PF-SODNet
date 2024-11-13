import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp
from torch.nn import  Parameter
from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized
from einops import rearrange
import pywt


##### basic ####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)


class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Chuncat(nn.Module):
    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1 + x2, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0] + x[1]


class Foldcut(nn.Module):
    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1 + x2


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True,
                 layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv, self).__init__()
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv1x1(self.conv_dw(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x


class RobustConv2(nn.Module):
    # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
    def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True,
                 layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv2, self).__init__()
        self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s,
                                              padding=0, bias=True, dilation=1, groups=1
                                              )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class Stem(nn.Module):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Stem, self).__init__()
        c_ = int(c2 / 2)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2 // 2, 3, k)
        self.cv3 = Conv(c1, c2 // 2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Bottleneck(nn.Module):
    # Darknet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Res, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class ResX(Res):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels


class Ghost(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(Ghost, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


##### end of basic #####


##### cspnet #####

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class GhostSPPCSPC(SPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class GhostStem(Stem):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, p, g, act)
        c_ = int(c2 / 2)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 3, 2)
        self.cv2 = GhostConv(c_, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 2)
        self.cv4 = GhostConv(2 * c_, c2, 1, 1)


class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResXCSPA(ResCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPB(ResCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPC(ResCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


##### end of cspnet #####


##### yolor #####

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


##### end of yolor #####


##### repvgg #####

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity,
                                                                        nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

            # print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        # print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        # print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepBottleneck(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)


class RepBottleneckCSPA(BottleneckCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPB(BottleneckCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPC(BottleneckCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepRes(Res):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResCSPA(ResCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPB(ResCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPC(ResCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResX(ResX):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResXCSPA(ResXCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPB(ResXCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPC(ResXCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


##### end of repvgg #####


##### transformer #####

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


##### end of transformer #####


##### yolov5 #####

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


##### end of yolov5 ######


##### orepa #####

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels,
                         kernel_size=self.conv.kernel_size,
                         stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation,
                         groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv


class OREPA_3x3_RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(OREPA_3x3_RepConv, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.branch_counter = 0

        self.weight_rbr_origin = nn.Parameter(
            torch.Tensor(out_channels, int(in_channels / self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1

        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            nn.init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg',
                                 torch.ones(kernel_size, kernel_size).mul(1.0 / kernel_size / kernel_size))
            self.branch_counter += 1

        else:
            raise NotImplementedError
        self.branch_counter += 1

        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(
                torch.zeros(in_channels, int(in_channels / self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels / self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels / self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)

        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(
                torch.Tensor(internal_channels_1x1_3x3, int(in_channels / self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(
            torch.Tensor(out_channels, int(internal_channels_1x1_3x3 / self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1

        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels * expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels * expand_ratio, 1, 1))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1

        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1

        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        self.fre_init()

        nn.init.constant_(self.vector[0, :], 0.25)  # origin
        nn.init.constant_(self.vector[1, :], 0.25)  # avg
        nn.init.constant_(self.vector[2, :], 0.0)  # prior
        nn.init.constant_(self.vector[3, :], 0.5)  # 1x1_kxk
        nn.init.constant_(self.vector[4, :], 0.5)  # dws_conv

    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels / 2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi * (h + 0.5) * (i + 1) / 3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi * (w + 0.5) * (i + 1 - half_fg) / 3)

        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):

        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])

        weight_rbr_avg = torch.einsum('oihw,o->oihw',
                                      torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg),
                                      self.vector[1, :])

        weight_rbr_pfir = torch.einsum('oihw,o->oihw',
                                       torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior),
                                       self.vector[2, :])

        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t / g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o / g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1,
                                              weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)

        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])

        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])

        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):

        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t / groups)
        i = int(ig * groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)

        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation,
                       groups=self.groups)

        return self.nonlinear(self.bn(out))


class RepConv_OREPA(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False,
                 use_se=False, nonlinear=nn.SiLU()):
        super(RepConv_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = c1
        self.out_channels = c2

        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert k == 3
        assert padding == 1

        padding_11 = padding - k // 2

        if nonlinear is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinear

        if use_se:
            self.se = SEBlock(self.out_channels, internal_neurons=self.out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k,
                                         stride=s,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=self.in_channels) if self.out_channels == self.in_channels and s == 1 else None
            self.rbr_dense = OREPA_3x3_RepConv(in_channels=self.in_channels, out_channels=self.out_channels,
                                               kernel_size=k, stride=s, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=s,
                                  padding=padding_11, groups=groups, dilation=1)
            print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3

        return self.nonlinearity(self.se(out))

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    # Not used for OREPA
    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1,
                                                                                                                   1, 1,
                                                                                                                   1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1,
                                                                                                             1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2,
                                            1:2] ** 2).sum()  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (
                    t3 ** 2 + t1 ** 2)).sum()  # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepConv):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        print(f"RepConv_OREPA.switch_to_deploy")
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation,
                                     groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')

        ##### end of orepa #####


##### swin transformer #####    

class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # print(attn.dtype, v.dtype)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            # print(attn.dtype, v.dtype)
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0, 'feature map h and w can not divide by window size'
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer(nn.Module):

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                                           shift_size=0 if (i % 2 == 0) else window_size // 2) for i in
                                      range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


##### end of swin transformer #####


##### swin transformer v2 ##### 

class WindowAttention_v2(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class Mlp_v2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_v2(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_v2(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer_v2(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #    # if window size is larger than input resolution, we don't partition windows
        #    self.shift_size = 0
        #    self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_v2(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size != 0 or W_ % self.window_size != 0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_v2(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinTransformer2Block(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=7):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size,
                                                              shift_size=0 if (i % 2 == 0) else window_size // 2) for i
                                      in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class ST2CSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        # self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class space_to_Depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class CBAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(CBAM, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Hardswish() if act else nn.Identity()
        self.ca = ChannelAttention(c2)
        self.sa = SpatialAttention()

    def forward(self, x):
        #      x = self.act(self.bn(self.conv(x)))
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


#    def fuseforward(self, x):
#      return self.act(self.conv(x))


##### end of swin transformer v2 #####

#         size_tensor = x.size()
#         return torch.cat([x[...,0:size_tensor[2]//2,0:size_tensor[3]//2],
#                          x[...,0:size_tensor[2]//2,size_tensor[3]//2:],
#                          x[...,size_tensor[2]//2:,0:size_tensor[3]//2],
#                          x[...,size_tensor[2]//2:,size_tensor[3]//2:]  ],1)

class v8_C2fBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(
            v8_C2fBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# class Cross_fusion(nn.Module):
#     def __init__(self):
#         super(Cross_fusion,self).__init__()
#         self.spd=space_to_Depth()
#
#
#
#
#     def forward(self,x):
class SRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2

        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # original size to 2x downsampling layer
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.cut_c = Cut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]

        # original size to 2x downsampling layer
        c = x  # c = [B, C/4, H, W]
        # CutD
        c = self.cut_c(c)  # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # ConvD
        x = self.conv_1(x)  # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2]
        x = self.conv_x1(x)  # x = [B, C/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # Concat + conv
        x = torch.cat([x, c], dim=1)  # x = [B, C, H/2, W/2]
        x = self.fusion1(x)  # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]

        # 2x to 4x downsampling layer
        r = x  # r = [B, C/2, H/2, W/2]
        x = self.conv_2(x)  # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]
        m = x  # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)  # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD
        m = self.max_m(m)  # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # CutD
        r = self.cut_r(r)  # r = [B, C, H/4, W/4]
        # Concat + conv
        x = torch.cat([x, r, m], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)  # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]
        return x  # x = [B, C, H/4, W/4]


# CutD
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)  # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


# Deep feature downsampling
class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):  # input: x = [B, C, H, W]
        c = x  # c = [B, C, H, W]
        x = self.conv(x)  # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x  # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)  # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)  # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)  # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x  # x = [B, 2C, H/2, W/2]


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


class HSRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=128):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2

        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # original size to 2x downsampling layer
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.haar_down = Down_wt(out_c14, out_c12)
        self.cut_c = HCut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels + out_c12, out_c12, kernel_size=1, stride=1)

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.haar_down_1 = Down_wt(out_c12, out_channels)
        self.cut_r = HCut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]

        # original size to 2x downsampling layer
        c = x  # c = [B, C/4, H, W]
        t = x  # t = [B, C/4, H, W]
        # Haar
        t = self.haar_down(t)  # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # CutD
        c = self.cut_c(c)  # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # ConvD
        x = self.conv_1(x)  # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2]
        x = self.conv_x1(x)  # x = [B, C/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # Concat + conv
        x = torch.cat([x, c, t], dim=1)  # x = [B, C, H/2, W/2]
        x = self.fusion1(x)  # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]

        # 2x to 4x downsampling layer
        r = x  # r = [B, C/2, H/2, W/2]
        f = x  # f = [B, C/2, H/2, W/2]
        x = self.conv_2(x)  # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]
        m = x  # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)  # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD
        m = self.max_m(m)  # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # Haar
        f = self.haar_down_1(f)
        # CutD
        r = self.cut_r(r)  # r = [B, C, H/4, W/4]
        # Concat + conv
        x = torch.cat([x, r, m, f], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)  # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]
        return x  # x = [B, C, H/4, W/4]


# CutD
class HCut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)  # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


# Deep feature downsampling
class HDRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = HCut(in_channels=in_channels, out_channels=out_channels)
        self.haar_down_1 = Down_wt(in_channels, out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(4 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):  # input: x = [B, C, H, W]
        c = x  # c = [B, C, H, W]
        h = x  # h = [B, C, H, W]
        x = self.conv(x)  # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x  # m = [B, 2C, H, W]

        # Haar
        h = self.haar_down_1(h)

        # CutD
        c = self.cut_c(c)  # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)  # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)  # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m, h], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)  # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x  # x = [B, 2C, H/2, W/2]


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)

        return x


class newEMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(newEMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.rfconv3x3 = RFAConv(channels // self.groups, channels // self.groups, kernel_size=3, stride=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x3 = self.sa(group_x)
        x1 = self.gn((group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()) * x3)
        x2 = self.rfconv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class New_EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(New_EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.sa((group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()))
        x1 = group_x * x1
        x1 = self.gn(x1)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        hw1 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        hw1_h = self.pool_h(hw1)
        hw1_w = self.pool_w(hw1).permute(0, 1, 3, 2)
        hw1_hw = self.conv1x1(torch.cat([hw1_h, hw1_w], dim=2))
        hw1_h, hw1_w = torch.split(hw1_hw, [h, w], dim=2)
        x1 = self.gn((hw1 * hw1_h.sigmoid() * hw1_w.permute(0, 1, 3, 2).sigmoid()))
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class Down_wtEMA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wtEMA, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ema = EMA(in_ch)

    def forward(self, x):
        x = self.ema(x)
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        # y_HL = yH[0]
        # y_LH = yH[1]
        # y_HH = yH[2]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # print(x.device)
        # print(x.dtype)
        x = torch.tensor(x)
        # print(x.dtype)
        # x = torch.as_tensor(x,dtype=torch.float32)
        # print(x.dtype)
        x = self.conv_bn_relu(x)
        # print(x)
        # print(x.device)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class NewDown_wtEMA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(NewDown_wtEMA, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ema = RFEMA(in_ch * 4)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.ema(x)
        x = self.conv_bn_relu(x)

        return x


class RFAConv(nn.Module):  # 基于Group Conv实现的RFAConv
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU())

    def forward(self, x):
        b, c = x.shape[0:2]
        # print(b,c)
        weight = self.get_weight(x)
        # print(weight.shape)
        h, w = weight.shape[2:]
        # print(h,w)
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        # print(weighted.shape)
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h,
                                                w)  # b c*kernel**2,h,w ->  b c k**2 h w   获得感受野空间特征
        # print(feature.shape)
        weighted_data = feature * weighted
        # print(weighted_data.shape)
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)

        # print(conv_data.shape)
        return self.conv(conv_data)


class RFEMA(nn.Module):
    def __init__(self, channels, kernel_size=3, c2=None, stride=1, factor=32):
        super(RFEMA, self).__init__()
        self.groups = factor
        self.kernel_size = kernel_size
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.generate_feature = nn.Sequential(
            nn.Conv2d(channels, channels * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=channels, bias=False),
            nn.BatchNorm2d(channels * (kernel_size ** 2)),
            nn.ReLU())
        self.conv = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=kernel_size,
                              stride=kernel_size)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        # mip = max(8, inp // reduction)
        # # self.bn1 = nn.BatchNorm2d(mip)
        # # self.act = h_swish()
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=kernel_size,
                                 padding=1)
        self.rfconv3x3 = RFAConv(channels // self.groups, channels // self.groups, kernel_size=3, stride=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        b, c = x.shape[0:2]
        # print(b,c)
        rfx = self.generate_feature(x)
        # print(rfx.size())
        h, w = rfx.shape[2:]
        # print(h,w)
        rfx = rearrange(rfx, 'b (c n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                        n2=self.kernel_size)
        # print(rfx.size())
        rfxb, rfxc, rfxh, rfxw = rfx.size()
        group_x = rfx.reshape(b * self.groups, -1, rfxh, rfxw)  # b*g,c//g,kh,kw
        # print(group_x.size())
        x_h = self.pool_h(group_x)
        # print(x_h.size())
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        x3 = self.sa(group_x)
        # print(x_w.size())
        cat1 = torch.cat([x_h, x_w], dim=2)
        # print(cat1.size)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # print(hw.size())
        x_h, x_w = torch.split(hw, [rfxh, rfxw], dim=2)
        # print(x_h.size())
        # print(x_w.size())
        # print(x_h.sigmoid().size())
        # print((group_x*x_h.sigmoid()).size())
        # print(x_w.permute(0, 1, 3, 2).sigmoid().size())

        # print(x3.size())
        # x1 = self.sa((group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()))
        # print(x1.size())
        x1 = self.conv(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid() * x3)
        # print(x1.size())
        data = x1
        x1 = self.gn(x1)
        # print(x1.size())
        # x1 = self.gn(x1)
        # print(x1.size())
        x2 = self.conv3x3(group_x)
        # print(x2.size())
        # print(self.agp(x1).size())
        # print(x2.size())
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # print(x11.size())
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # print(x12.size())
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # print(x21.size())
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # print(x22.size())
        # print(torch.matmul(x11, x12).size())
        # print(torch.matmul(x21, x22).size())
        # print((torch.matmul(x11, x12) + torch.matmul(x21, x22)).size())
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # print(weights.size())
        return (data * weights.sigmoid()).reshape(b, c, h, w)


class RFMCA(nn.Module):
    def __init__(self, channels, kernel_size=3, c2=None, stride=1, factor=32):
        super(RFMCA, self).__init__()
        self.groups = factor
        self.kernel_size = kernel_size
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.bn1 = nn.BatchNorm2d(channels // self.groups)
        self.act = nn.ReLU()
        self.generate_feature = nn.Sequential(
            nn.Conv2d(channels, channels * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=channels, bias=False),
            nn.BatchNorm2d(channels * (kernel_size ** 2)),
            nn.ReLU())
        self.conv = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=kernel_size,
                              stride=kernel_size)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=kernel_size,
                                 padding=1)
        self.sa = SpatialAttention()

    def forward(self, x):
        b, c = x.shape[0:2]
        # print(b,c)
        rfx = self.generate_feature(x)
        # print(rfx.size())
        h, w = rfx.shape[2:]
        # print(h,w)
        rfx = rearrange(rfx, 'b (c n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                        n2=self.kernel_size)
        # print(rfx.size())
        rfxb, rfxc, rfxh, rfxw = rfx.size()
        group_x = rfx.reshape(b * self.groups, -1, rfxh, rfxw)  # b*g,c//g,kh,kw
        # print(group_x.size())
        x_h = self.pool_h(group_x)
        # print(x_h.size())
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        x3 = self.sa(group_x)
        # print(x_w.size())
        cat1 = torch.cat([x_h, x_w], dim=2)
        # print(cat1.size)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        # print(hw.size())
        hw = self.bn1(hw)
        hw = self.act(hw)
        x_h, x_w = torch.split(hw, [rfxh, rfxw], dim=2)
        # print(x3.size())
        # x1 = self.sa((group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()))
        # print(x1.size())
        x1 = self.conv(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid() * x3)
        # print(x1.size())
        return x1.reshape(b, c, h, w)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class MyDown_wtMCA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MyDown_wtMCA, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.mca = RFMCA(in_ch * 4)

    def forward(self, x):
        # print(x.size())
        yL, yH = self.wt(x)
        # print(yL.size())
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        # print(x.size())
        x = self.mca(x)
        x = self.conv_bn_relu(x)
        return x


class All_EMA(nn.Module):
    def __init__(self, inchannels, outchannels, c2=None, factor=32):
        super(All_EMA, self).__init__()
        self.groups = factor
        assert inchannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.bn = nn.BatchNorm2d(inchannels // self.groups)
        self.act = h_swish()
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(inchannels // self.groups, inchannels // self.groups)
        self.conv1x1 = nn.Conv2d(inchannels // self.groups, outchannels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(inchannels // self.groups, outchannels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        hw = self.bn(hw)
        hw = self.act(hw)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gca1 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        gca1_h = self.pool_h(gca1)
        gca1_w = self.pool_w(gca1).permute(0, 1, 3, 2)
        gca1_hw = self.conv1x1(torch.cat([gca1_h, gca1_w], dim=2))
        gca1_hw = self.bn(gca1_hw)
        gca1_hw = self.act(gca1_hw)
        gca1_hw_h, gca1_hw_w = torch.split(gca1_hw, [h, w], dim=2)
        out = group_x * gca1_hw_h.sigmoid() * gca1_hw_w.permute(0, 1, 3, 2).sigmoid()
        # out1 = self.conv3x3(group_x)
        # out = out+out1
        return out.reshape(b, c, h, w)


class TU_Down_wtEMA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TU_Down_wtEMA, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.ema = All_EMA(out_ch)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        x = self.ema(x)
        return x


class NEW_All_EMA(nn.Module):
    def __init__(self, inchannels, outchannels, c2=None, factor=32):
        super(NEW_All_EMA, self).__init__()
        self.groups = factor
        assert inchannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.bn = nn.BatchNorm2d(inchannels // self.groups)
        self.act = h_swish()
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(inchannels // self.groups, inchannels // self.groups)
        self.con1 = nn.Conv2d(inchannels // self.groups, inchannels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(inchannels // self.groups, outchannels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(inchannels // self.groups, outchannels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.con1(torch.cat([x_h, x_w], dim=2))
        hw = self.bn(hw)
        hw = self.act(hw)
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gca1 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        gca1 = self.conv1x1(gca1)
        gca1_h = self.pool_h(gca1)
        gca1_w = self.pool_w(gca1).permute(0, 1, 3, 2)
        gca1_hw = self.conv1(torch.cat([gca1_h, gca1_w], dim=2))
        gca1_hw = self.bn(gca1_hw)
        gca1_hw = self.act(gca1_hw)
        gca1_hw_h, gca1_hw_w = torch.split(gca1_hw, [h, w], dim=2)
        out = group_x * gca1_hw_h.sigmoid() * gca1_hw_w.permute(0, 1, 3, 2).sigmoid()
        out = out.reshape(b, c, h, w)
        # out1 = self.conv3x3(group_x)
        # out = out+out1
        return out.reshape(b, c, h, w)


class TU_EMA(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=32):
        super(TU_EMA, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gca1 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        gca1_h = self.pool_h(gca1)
        gca1_w = self.pool_w(gca1).permute(0, 1, 3, 2)
        gca1_hw = self.conv1x1(torch.cat([gca1_h, gca1_w], dim=2))
        gca1_h, gca1_w = torch.split(gca1_hw, [h, w], dim=2)
        x1 = self.gn(group_x * gca1_h.sigmoid() * gca1_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        out = self.con1(out)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out


class O_EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(O_EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class TU_Down_wtEMA1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TU_Down_wtEMA1, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
        )
        # self.ema = All_EMA(out_ch)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        x = torch.cat([yL, x], dim=1)
        return x


class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class TU_Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(TU_Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
        )
        # self.ema = All_EMA(out_ch)
        self.ca = ECA(in_ch * 3)

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([y_HL, y_LH, y_HH], dim=1)
        # x = self.ca(x)
        x = self.conv_bn_relu(x)
        x = torch.cat([yL, x], dim=1)
        return x


class MY_Down_dwt_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MY_Down_dwt_conv, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d((in_ch // 2) * 3, (in_ch // 2) * 3, kernel_size=1, stride=1),
            nn.BatchNorm2d((in_ch // 2) * 3),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
        )
        # self.ca = ECA(in_ch*3)
        self.conv_x = nn.Conv2d(in_ch // 2, out_ch // 4, kernel_size=3, stride=2, padding=1)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(in_ch // 2)

    def forward(self, x):
        x = self.conv_bn_relu1(x)
        y = x
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        y = self.conv_x(y)
        y = self.batch_norm_x(y)
        y = self.act_x(y)
        # print(x.size())
        z = torch.add(yL, y)
        x = torch.cat([x, z], dim=1)
        return x


class TU_EMA_sppc(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=32):
        super(TU_EMA_sppc, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, outChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        # self.con1 = nn.Conv2d(inChannels ,outChannels, kernel_size=1, stride=1, padding=0)
        # self.con2 = Conv(inChannels,outChannels,1,1,0)
        # self.con1 = Conv(inChannels,outChannels,k=1,s=1,p=0)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gca1 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        gca1_h = self.pool_h(gca1)
        gca1_w = self.pool_w(gca1).permute(0, 1, 3, 2)
        gca1_hw = self.conv1x1(torch.cat([gca1_h, gca1_w], dim=2))
        gca1_h, gca1_w = torch.split(gca1_hw, [h, w], dim=2)
        x1 = self.gn(group_x * gca1_h.sigmoid() * gca1_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        # out = self.con1(out)
        return out


class TU_EMA2(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=32):
        super(TU_EMA2, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_h1 = self.maxpool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.pool_w(group_x)
        x_w1 = self.maxpool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1))).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h2, x_w2], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gca1 = group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid()
        gca1_h = self.pool_h(gca1)
        gca1_h1 = self.maxpool_h(gca1)
        gca1_h2 = self.nconv1x1(torch.cat([gca1_h, gca1_h1], dim=1))
        gca1_w = self.pool_w(gca1)
        gca1_w1 = self.maxpool_w(gca1)
        gca1_w2 = (self.nconv1x1(torch.cat([gca1_w, gca1_w1], dim=1))).permute(0, 1, 3, 2)
        gca1_hw = self.conv1x1(torch.cat([gca1_h2, gca1_w2], dim=2))
        gca1_h, gca1_w = torch.split(gca1_hw, [h, w], dim=2)
        x1 = self.gn(group_x * gca1_h.sigmoid() * gca1_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        out = self.con1(out)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out


class attention_dwt_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(attention_dwt_conv, self).__init__()
        self.splictPatch = nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, stride=1),
                                         nn.BatchNorm2d(in_ch // 2),
                                         nn.ReLU(inplace=True),
                                         )
        self.attention = TU_EMA(inChannels=in_ch // 2, outChannels=in_ch // 2)
        # self.upsample = nn.Upsample(scale_factor=1,mode="bilinear",align_corners=False)
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid()
        )
        self.avg = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = x
        x = self.splictPatch(x)
        x = self.attention(x)
        # x = self.upsample(x)
        x = self.conv_bn_relu1(x)
        y = y * x
        result = self.avg(y)
        result = self.conv_bn_relu2(result)
        return result


class FFDD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(FFDD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.p = nn.Sequential(nn.Conv2d(in_ch, out_ch // 2, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d(out_ch // 2),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(out_ch // 2, (out_ch // 2) * 3, kernel_size=1, stride=1),
                               nn.Tanh()
                               )
        self.u = nn.Sequential(nn.Conv2d((out_ch // 2) * 3, (out_ch // 2) * 3, kernel_size=3, stride=1, padding=1),
                               nn.BatchNorm2d((out_ch // 2) * 3),
                               nn.ReLU(inplace=True),
                               nn.Conv2d((out_ch // 2) * 3, in_ch, kernel_size=1, stride=1),
                               nn.Tanh())
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(in_ch * 3, out_ch // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch // 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([y_HL, y_LH, y_HH], dim=1)
        yL_p = self.p(yL)
        x_p = x - yL_p
        x_u = self.u(x_p)
        yL_u = yL + x_u
        x = self.conv_bn_relu1(x)
        result = torch.cat([yL_u, x], dim=1)
        return result


class TU_EMA3(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=32):
        super(TU_EMA3, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxpool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxpool_w = nn.AdaptiveMaxPool2d((1, None))
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_h1 = self.maxpool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.pool_w(group_x)
        x_w1 = self.maxpool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1))).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h2, x_w2], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        gca1 = group_x * x_h.sigmoid() * (x_w.permute(0, 1, 3, 2).sigmoid())
        gca1_h = self.pool_h(gca1)
        gca1_w = self.pool_w(gca1).permute(0, 1, 3, 2)
        gca1_hw = self.conv1x1(torch.cat([gca1_h, gca1_w], dim=2))
        gca1_h, gca1_w = torch.split(gca1_hw, [h, w], dim=2)
        x1 = self.gn(group_x * gca1_h.sigmoid() * gca1_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        out = self.con1(out)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out


class TU_EMA5(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=32):
        super(TU_EMA5, self).__init__()
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels, inChannels, kernel_size=7, padding=7 // 2, groups=inChannels, bias=False)
        self.gn = nn.GroupNorm(16, inChannels)
        self.conv3x3 = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=1, padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        y = x
        x_h = self.pool_h(x).reshape(b, c, h)
        x_w = self.pool_w(x).reshape(b, c, w)
        x_h = self.gn(x_h)
        x_w = self.gn(x_w)
        x_h = x_h.sigmoid().reshape(b, c, h, 1)
        x_w = x_w.sigmoid().reshape(b, c, 1, w)
        x1 = x * x_h * x_w
        x1_h = self.pool_h(x1).reshape(b, c, h)
        x1_w = self.pool_w(x1).reshape(b, c, w)
        x1_h = self.gn(x1_h)
        x1_w = self.gn(x1_w)
        x1_h = x1_h.sigmoid().reshape(b, c, h, 1)
        x1_w = x1_w.sigmoid().reshape(b, c, 1, w)
        x2 = y * x1_h * x1_w
        # x3 = self.conv3x3(x)
        # b2, c2, h2, w2 = x2.size()
        # x21 = x2.reshape(b2,c2,-1)
        # x31 = x3.reshape(b2,c2,-1)
        # x22 = self.softmax(self.agp(x2).reshape(b2,-1,1)).permute(0, 2, 1)
        # x32 = self.softmax(self.agp(x3).reshape(b2,-1,1)).permute(0, 2, 1)
        # weights = (torch.matmul(x22, x31) + torch.matmul(x32, x21)).reshape(b,-1,h,w)
        # out = (x * weights.sigmoid())
        out = self.con1(x2)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out


class TU_EMA6(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=16):
        super(TU_EMA6, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=7, padding=7 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(factor, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,
                                                                                                    1, w)
        gca1_h = self.avgPool_h(gca1)
        gca1_h1 = self.maxPool_h(gca1)
        gca1_h2 = self.nconv1x1(torch.cat([gca1_h, gca1_h1], dim=1))
        gca1_w = self.avgPool_w(gca1)
        gca1_w1 = self.maxPool_w(gca1)
        gca1_w2 = (self.nconv1x1(torch.cat([gca1_w, gca1_w1], dim=1)))
        gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups, -1, h)))
        gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups, -1, w)))
        x1 = group_x * gca1_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * gca1_w2.sigmoid().reshape(b * self.groups,
                                                                                                        -1, 1, w)
        # x2 = self.conv3x3(group_x)
        # x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = x1.reshape(b, c, h, w)
        out = self.con1(out)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out


class TU_EMA7(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=16):
        super(TU_EMA7, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=7, padding=7 // 2,
                                bias=False)
        # groups=inChannels // self.groups,
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)
        self.c_down = Conv(inChannels, outChannels, k=1, s=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = x_h + x_h1
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = x_w +x_w1
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,1, w)
        gca1_h = self.avgPool_h(gca1)
        gca1_h1 = self.maxPool_h(gca1)
        gca1_h2 = gca1_h +gca1_h1
        gca1_w = self.avgPool_w(gca1)
        gca1_w1 = self.maxPool_w(gca1)
        gca1_w2 = gca1_w + gca1_w1
        gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups, -1, h)))
        gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups, -1, w)))
        x1 = group_x * gca1_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * gca1_w2.sigmoid().reshape(b * self.groups, -1, 1, w)
        out = x1.reshape(b, c, h, w)
        out = self.c_down(out)
        return out


class TU_EMA7_1(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=16):
        super(TU_EMA7_1, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=7, padding=7 // 2,
                                bias=False)
        # groups=inChannels // self.groups,
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)
        self.c_down = Conv(inChannels, outChannels, k=1, s=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        # x_h2 = self.nconv1x1(torch.cat([x_h,x_h1],dim=1))
        x_h2 = x_h + x_h1
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        # x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        x_w2 = x_w +x_w1
        # print(self.conv1d(x_h2.reshape(b * self.groups,-1,h)).size())
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        # x_h2 = self.gn(self.conv1d(x_h.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        # x_w2 = self.gn(self.conv1d(x_w.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,1, w)
        gca1_h = self.avgPool_h(gca1)
        gca1_h1 = self.maxPool_h(gca1)
        # gca1_h2 = self.nconv1x1(torch.cat([gca1_h, gca1_h1], dim=1))
        gca1_h2 = gca1_h +gca1_h1
        gca1_w = self.avgPool_w(gca1)
        gca1_w1 = self.maxPool_w(gca1)
        # gca1_w2 = (self.nconv1x1(torch.cat([gca1_w, gca1_w1], dim=1)))
        gca1_w2 = gca1_w + gca1_w1
        gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups, -1, h)))
        # gca1_h2 = self.gn(self.conv1d(gca1_h.reshape(b * self.groups, -1, h)))
        gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups, -1, w)))
        # gca1_w2 = self.gn(self.conv1d(gca1_w.reshape(b * self.groups, -1, w)))
        x1 = group_x * gca1_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * gca1_w2.sigmoid().reshape(b * self.groups, -1, 1, w)
        # x2 = self.conv3x3(group_x)
        # x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = x1.reshape(b, c, h, w)
        # out = self.con1(out)
        # out = self.batch_norm_x(out)
        # out = self.act_x(out)
        out = self.c_down(out)
        return out


class TU_EMA8(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=16):
        super(TU_EMA8, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=7, padding=7 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        # print(self.conv1d(x_h2.reshape(b * self.groups,-1,h)).size())
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,
                                                                                                    1, w)
        # gca1_h = self.avgPool_h(gca1)
        # gca1_h1 = self.maxPool_h(gca1)
        # gca1_h2 = self.nconv1x1(torch.cat([gca1_h,gca1_h1],dim=1))
        # gca1_w = self.avgPool_w(gca1)
        # gca1_w1 = self.maxPool_w(gca1)
        # gca1_w2 = (self.nconv1x1(torch.cat([gca1_w ,gca1_w1],dim=1)))
        # gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups,-1,h)))
        # gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups,-1,w)))
        # x1 = group_x * gca1.sigmoid().reshape(b * self.groups,-1,h,1) * gca1_w2.sigmoid().reshape(b * self.groups,-1,1,w)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(gca1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = gca1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # out = (gca1 * weights.sigmoid()).reshape(b, c, h, w)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        # out = self.con1(out)
        # out = self.batch_norm_x(out)
        # out = self.act_x(out)
        return out


class TU_EMA9(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=16):
        super(TU_EMA9, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=7, padding=7 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        # print(self.conv1d(x_h2.reshape(b * self.groups,-1,h)).size())
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,
                                                                                                    1, w)
        # gca1_h = self.avgPool_h(gca1)
        # gca1_h1 = self.maxPool_h(gca1)
        # gca1_h2 = self.nconv1x1(torch.cat([gca1_h,gca1_h1],dim=1))
        # gca1_w = self.avgPool_w(gca1)
        # gca1_w1 = self.maxPool_w(gca1)
        # gca1_w2 = (self.nconv1x1(torch.cat([gca1_w ,gca1_w1],dim=1)))
        # gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups,-1,h)))
        # gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups,-1,w)))
        # x1 = group_x * gca1.sigmoid().reshape(b * self.groups,-1,h,1) * gca1_w2.sigmoid().reshape(b * self.groups,-1,1,w)
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(gca1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = gca1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # out = (gca1 * weights.sigmoid()).reshape(b, c, h, w)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        out = self.con1(out)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out


class TU_EMA11(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=8):
        super(TU_EMA11, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=5, padding=5 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        x = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1, 1,
                                                                                                 w)
        out = x.reshape(b, c, h, w)
        # out = self.con1(out)
        # out = self.batch_norm_x(out)
        # out = self.act_x(out)
        return out


class TU_EMA12(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=8):
        super(TU_EMA12, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=5, padding=5 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inChannels // self.groups),
            nn.ReLU(inplace=True),
        )
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = self.conv_bn_relu1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = (self.conv_bn_relu1(torch.cat([x_w, x_w1], dim=1)))
        # print(self.conv1d(x_h2.reshape(b * self.groups,-1,h)).size())
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,
                                                                                                    1, w)
        gca1_h = self.avgPool_h(gca1)
        gca1_h1 = self.maxPool_h(gca1)
        gca1_h2 = self.nconv1x1(torch.cat([gca1_h, gca1_h1], dim=1))
        gca1_w = self.avgPool_w(gca1)
        gca1_w1 = self.maxPool_w(gca1)
        gca1_w2 = (self.nconv1x1(torch.cat([gca1_w, gca1_w1], dim=1)))
        gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups, -1, h)))
        gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups, -1, w)))
        x1 = group_x * gca1_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * gca1_w2.sigmoid().reshape(b * self.groups,
                                                                                                        -1, 1, w)
        # x2 = self.conv3x3(group_x)
        # x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        # out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        out = x1.reshape(b, c, h, w)
        # out = self.con1(out)
        # out = self.batch_norm_x(out)
        # out = self.act_x(out)
        return out


class CBAMLayer(nn.Module):
    def __init__(self, channel, oChannel, reduction=16, spatial_kernel=3):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv(channel, oChannel, )

    def forward(self, x):
        # print(x.size())
        max_out = self.mlp(self.max_pool(x))
        # print(max_out.size())
        avg_out = self.mlp(self.avg_pool(x))
        # print(avg_out.size())
        channel_out = self.sigmoid(max_out + avg_out)

        x = channel_out * x
        # print(x.size())

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(max_out.size())
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(avg_out.size())
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print(spatial_out.size())
        x = spatial_out * x
        # print(x.size())
        x = self.conv1(x)
        return x


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class CBAMLayer_new(nn.Module):
    def __init__(self, channel, oChannel, reduction=16, spatial_kernel=3):
        super(CBAMLayer_new, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv(channel, oChannel, )

    def forward(self, x):
        y = x
        # print(x.size())
        max_out = self.mlp(self.max_pool(x))
        # print(max_out.size())
        avg_out = self.mlp(self.avg_pool(x))
        # print(avg_out.size())
        channel_out = self.sigmoid(max_out + avg_out)

        x = channel_out * x
        # print(x.size())

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(max_out.size())
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(avg_out.size())
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print(spatial_out.size())
        x = spatial_out * x
        # print(x.size())
        # x = self.conv1(x)
        x = y + x
        return x


class CBAMLayer_new1(nn.Module):
    def __init__(self, channel, oChannel, reduction=16, spatial_kernel=3):
        super(CBAMLayer_new1, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)

            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv(channel, oChannel, )

    def forward(self, x):
        y = x
        # print(x.size())
        max_out = self.mlp(self.max_pool(x))
        # print(max_out.size())
        avg_out = self.mlp(self.avg_pool(x))
        # print(avg_out.size())
        channel_out = self.sigmoid(max_out + avg_out)

        x = channel_out * x
        # print(x.size())

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print(max_out.size())
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print(avg_out.size())
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print(spatial_out.size())
        x = spatial_out * x
        # print(x.size())
        x = y + x
        x = self.conv1(x)
        return x


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()  # 调用父类构造函数
        assert c_num >= group_num  # 断言 c_num 大于等于 group_num
        self.group_num = group_num  # 设置分组数量
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))  # 创建可训练参数 gamma
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))  # 创建可训练参数 beta
        self.eps = eps  # 设置小的常数 eps 用于稳定计算

    def forward(self, x):
        N, C, H, W = x.size()  # 获取输入张量的尺寸
        x = x.view(N, self.group_num, -1)  # 将输入张量重新排列为指定的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每个组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每个组的标准差
        x = (x - mean) / (std + self.eps)  # 应用批量归一化
        x = x.view(N, C, H, W)  # 恢复原始形状
        # print(x.size())
        # print(self.gamma.size())
        # print(self.beta.size())
        return x * self.gamma + self.beta  # 返回归一化后的张量


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    # https://github.com/cheng-haha/ScConv/blob/main/ScConv.py
    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


class MyIdea(nn.Module):
    def __init__(self, inputChannels, outChannels):
        super(MyIdea, self).__init__()
        self.ps = nn.PixelShuffle(2)
        # self.conv_bn_relu2 = nn.Sequential(
        #     nn.Conv2d(inputChannels / 4, outChannels, kernel_size=1, stride=1),
        #     nn.BatchNorm2d(outChannels),
        #     nn.ReLU(inplace=True)
        # )

        # self.sa = SpatialAttention()
        # self.conv1x1 = Conv(int(inputChannels / 4), int(outChannels), k=1, s=1, p=0)
        self.conv2x2 = Conv(int(outChannels), int(outChannels), k=2, s=2, p=0)

    def forward(self, x):
        # weight = self.sa(x)
        # print(weight.size())
        ps = self.ps(x)
        x = self.conv2x2(ps)
        # x = x * weight
        return x


class TU_TU(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=8):
        super(TU_TU, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=5, padding=5 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.act_x = nn.ReLU(inplace=True)
        self.batch_norm_x = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        x_h2 = self.nconv1x1(torch.cat([x_h, x_h1], dim=1))
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        x = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1, 1,
                                                                                                 w)
        out = x.reshape(b, c, h, w)
        out = self.con1(out)
        out = self.batch_norm_x(out)
        out = self.act_x(out)
        return out
class ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self,inChannnels,outChannels):
        super(ChannelAttentionModule, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.conv1x1 = nn.Conv2d(inChannnels,outChannels, 1, 1, 0)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        batch_size, c, height, width = x.size()
        new_x = self.conv1x1(x)
        b,oc, h,w=new_x.size()

        y = new_x.view(b, oc, -1)#c’*hw
        feat_x_transpose = x.view(batch_size, c, -1).permute(0, 2, 1)#HW*c
        attention_y = torch.bmm(y, feat_x_transpose)#c'*c
        attention_new = torch.max(attention_y, dim=-1, keepdim=True)[0].expand_as(attention_y) - attention_y#c'*c 缩小后的值
        attention_y = self.softmax(attention_new) #c'*c

        feat_a = x.view(batch_size, c, height * width)#c*HW
        new_y = torch.bmm(attention_y, feat_a)#c'*hw
        new_y = new_y.view(batch_size, oc,  h, w)#H*W*c'
        out = self.gamma*new_y + new_x #
        out = self.gamma*new_y + new_x #
        return out



class new_TU_EMA7(nn.Module):
    def __init__(self, inChannels, outChannels, c2=None, factor=16):
        super(new_TU_EMA7, self).__init__()
        self.groups = factor
        assert inChannels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.avgPool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.avgPool_w = nn.AdaptiveAvgPool2d((1, None))
        self.maxPool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.maxPool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv1d = nn.Conv1d(inChannels // self.groups, inChannels // self.groups, kernel_size=7, padding=7 // 2,
                                groups=inChannels // self.groups, bias=False)
        self.gn = nn.GroupNorm(inChannels // self.groups, inChannels // self.groups)
        self.conv1x1 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=1, stride=1,
                                 padding=0)
        self.nconv1x1 = nn.Conv2d((inChannels // self.groups) * 2, inChannels // self.groups, kernel_size=1, stride=1,
                                  padding=0)
        self.conv3x3 = nn.Conv2d(inChannels // self.groups, inChannels // self.groups, kernel_size=3, stride=1,
                                 padding=1)
        self.con1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        # self.act_x = nn.ReLU(inplace=True)
        # self.batch_norm_x = nn.BatchNorm2d(outChannels)
        self.c_down = Conv(inChannels, outChannels, k=1, s=1)

    def forward(self, x):
        b, c, h, w = x.size()
        y = x
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.avgPool_h(group_x)
        x_h1 = self.maxPool_h(group_x)
        # x_h2 = self.nconv1x1(torch.cat([x_h,x_h1],dim=1))
        x_h2 = x_h + x_h1
        x_w = self.avgPool_w(group_x)
        x_w1 = self.maxPool_w(group_x)
        # x_w2 = (self.nconv1x1(torch.cat([x_w, x_w1], dim=1)))
        x_w2 = x_w +x_w1
        # print(self.conv1d(x_h2.reshape(b * self.groups,-1,h)).size())
        x_h2 = self.gn(self.conv1d(x_h2.reshape(b * self.groups, -1, h)))
        x_w2 = self.gn(self.conv1d(x_w2.reshape(b * self.groups, -1, w)))
        gca1 = group_x * x_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * x_w2.sigmoid().reshape(b * self.groups, -1,1, w)
        gca1_h = self.avgPool_h(gca1)
        gca1_h1 = self.maxPool_h(gca1)
        # gca1_h2 = self.nconv1x1(torch.cat([gca1_h, gca1_h1], dim=1))
        gca1_h2 = gca1_h +gca1_h1
        gca1_w = self.avgPool_w(gca1)
        gca1_w1 = self.maxPool_w(gca1)
        # gca1_w2 = (self.nconv1x1(torch.cat([gca1_w, gca1_w1], dim=1)))
        gca1_w2 = gca1_w + gca1_w1
        gca1_h2 = self.gn(self.conv1d(gca1_h2.reshape(b * self.groups, -1, h)))
        gca1_w2 = self.gn(self.conv1d(gca1_w2.reshape(b * self.groups, -1, w)))
        x1 = group_x * gca1_h2.sigmoid().reshape(b * self.groups, -1, h, 1) * gca1_w2.sigmoid().reshape(b * self.groups, -1, 1, w)
        # x2 = self.conv3x3(group_x)
        # x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        # x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        # weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        out = x1.reshape(b, c, h, w)
        # out = self.con1(out)
        # out = self.batch_norm_x(out)
        # out = self.act_x(out)
        # out = self.c_down(out)
        out = y + out
        return out