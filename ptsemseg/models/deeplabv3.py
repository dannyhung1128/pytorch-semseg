#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import _ConvBatchNormReLU, _ResBlock, _ResBlockMG
from mobilenetv2 import MobileNetV2
from torch.nn import Parameter


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            'c0',
            _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1),
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                'c{}'.format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(1)),
                ('conv', _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
            ])
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.upsample(h, size=x.shape[2:], mode='bilinear')]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Sequential):
    """DeepLab v3"""

    def __init__(self, n_classes, n_blocks, pyramids, multi_grid=[1, 2, 1]):
        super(DeepLabV3, self).__init__()
       
        self.mobilenetv2 = MobileNetV2()
        self.add_module('aspp', _ASPPModule(1280, 256, pyramids))
        self.add_module('fc1', _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1))
        self.add_module('fc2', nn.Conv2d(256, n_classes, kernel_size=1))

    def forward(self, x):
        x = self.mobilenetv2(x)
        x = self.aspp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x 
    
    def loadMobileNetv2(self, path='../../mobilenetv2_718.pth.tar'):
        state_dict = torch.load(path)
        self.mobilenetv2.load_state_dict(state_dict)
    
    def load_pretrained_model(self, path='deeplabv3_cityscapes_24_best_model.pkl'):
        state_dict = torch.load(path)['model_state']
        for name, param in state_dict.items():
            if name[13:] not in self.state_dict().keys():
                continue
            if isinstance(param, Parameter):
                param = param.data
            print(name)
            self.state_dict()[name[13:]].copy_(param)

if __name__ == '__main__':
    model = DeepLabV3(n_classes=3, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18])
    model.eval()
    print list(model.named_children())
    image = torch.autograd.Variable(torch.randn(1, 3, 513, 513), volatile=True)
    print model(image)[0].size()
