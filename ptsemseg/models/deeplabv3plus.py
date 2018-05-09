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
from deeplabv3 import _ASPPModule
from ptsemseg.loss import *


class DeepLabV3Plus(nn.Sequential):
    """DeepLab v3+ (OS=8)"""

    def __init__(self, n_classes, n_blocks, pyramids, multi_grid=[1, 2, 1]):
        super(DeepLabV3Plus, self).__init__()
        self.add_module(
            'layer1',
            nn.Sequential(
                OrderedDict([
                    ('conv1', _ConvBatchNormReLU(3, 64, 7, 1, 2, 1)),
                    ('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                ])
            )
        )
        self.add_module('layer2', _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))  # output_stride=4
        self.add_module('layer3', _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))  # output_stride=8
        self.add_module('layer4', _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))  # output_stride=8
        self.add_module('layer5', _ResBlockMG(n_blocks[3], 1024, 512, 2048, 1, 2, mg=multi_grid))
        self.add_module('aspp', _ASPPModule(2048, 256, pyramids))
        self.add_module('fc1', _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1))
        self.add_module('reduce', _ConvBatchNormReLU(512, 48, 1, 1, 0, 1))
        self.add_module(
            'fc2',
            nn.Sequential(
                OrderedDict([
                    ('conv1', _ConvBatchNormReLU(304, 256, 3, 1, 1, 1)),
                    ('conv2', _ConvBatchNormReLU(256, 256, 3, 1, 1, 1)),
                    ('conv3', nn.Conv2d(256, n_classes, kernel_size=1)),
                ])
            )
        )

        # Define auxiliary loss function
        self.loss = multi_scale_cross_entropy2d


    def forward(self, x):
        #print("0", x.shape)
        h = self.layer1(x)
        #print("1", h.shape)
        h = self.layer2(h)
        #print("2", h.shape)
        h = self.layer3(h)
        #print("3", h.shape)
        h_ = self.reduce(h)
        #print("h_", h_.shape)
        h = self.layer4(h)
        #print("4", h.shape)
        h = self.layer5(h)
        #print("5", h.shape)
        
        h = self.aspp(h)
        #print("aspp", h.shape)
        
        h = self.fc1(h)
        #print("fc1", h.shape)
        
        h = F.upsample(h, size=h_.shape[2:], mode='bilinear')
        #print("up", h.shape)
        
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        #print("fc2", h.shape)
        h = F.upsample(h, scale_factor=4, mode='bilinear')
        #print("opt", h.shape)
        return h

    def freeze_bn(self):
        for m in self.named_modules():
            if 'layer' in m[0]:
                if isinstance(m[1], nn.BatchNorm2d):
                    m[1].eval()


if __name__ == '__main__':
    model = DeepLabV3Plus(n_classes=21, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18])
    model.freeze_bn()
    model.eval()
    print list(model.named_children())
    image = torch.autograd.Variable(torch.randn(1, 3, 513, 513), volatile=True)
    print model(image)[0].size()
