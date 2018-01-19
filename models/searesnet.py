#!/usr/bin/env python

"""
    searesnet.py
"""

import sys
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

sys.path.append('..')
import morph_layers as mm
from seanet import SeaNet

# EPS = 1e-5
# def make_seablock(in_planes, planes, stride=1, input_dim=32):
#     if stride == 1:
#         return SeaNet({
#             1: (mm.MorphBatchNorm2d(in_planes, relu=True, eps=EPS), 0),
#             2: (mm.MorphConv2d(in_planes, planes, kernel_size=3, padding=1, stride=1, bias=False), 1),
#             3: (mm.MorphBatchNorm2d(planes, relu=True, eps=EPS), 2),
#             4: (mm.MorphConv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False), 3),
#             5: (mm.AddLayer(alpha=0.5), [4, 0])
#         }, input_shape=(in_planes, input_dim, input_dim), tags='simple')
#     else:
#         return SeaNet({
#             1: (mm.MorphBatchNorm2d(in_planes, relu=True, eps=EPS), 0),
#             2: (mm.MorphConv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False), 1),
#             3: (mm.MorphBatchNorm2d(planes, relu=True, eps=EPS), 2),
#             4: (mm.MorphConv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False), 3),
#             5: (mm.MorphConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), 1), # shortcut
#             6: (mm.AddLayer(alpha=0.5), [5, 4]),
#         }, input_shape=(in_planes, input_dim, input_dim), tags='downsample')



def make_simple_seablock(in_planes, planes, stride=1, input_dim=32):
    if stride == 1:
        # simple
        return SeaNet({
            1: (mm.MorphConv2d(in_planes, planes, kernel_size=3, padding=1, stride=1, bias=False), 0),
            2: (mm.AddLayer(alpha=0.5), [1, 0]),
            3: (mm.IdentityLayer(), 2), # To allow layers to be added at the end
        }, input_shape=(in_planes, input_dim, input_dim), tags='simple')
    else:
        # downsample
        return SeaNet({
            1: (mm.MorphConv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False), 0),
            2: (mm.MorphConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), 0),
            3: (mm.AddLayer(alpha=0.5), [2, 1]),
            4: (mm.IdentityLayer(), 3), # To allow layers to be added at the end
        }, input_shape=(in_planes, input_dim, input_dim), tags='downsample')



class SeaResNet(nn.Module):
    def __init__(self, num_blocks=[2, 2, 2, 2], num_classes=10, input_shape=(3, 32, 32)):
        super(SeaResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.in_planes = 64
        self.input_dim = input_shape[-1]
        self.core_layers = nn.Sequential(*[
            self._make_layer(64, num_blocks[0], stride=1),
            self._make_layer(128, num_blocks[1], stride=2),
            self._make_layer(256, num_blocks[2], stride=2),
            self._make_layer(512, num_blocks[3], stride=2),
        ])
        
        self.linear = nn.Linear(512, num_classes)
        
        self._input_shape = input_shape
        self._input_data  = Variable(torch.randn((10,) + self._input_shape))
        
        self._sea_blocks = defaultdict(list)
        for child1 in self.core_layers.children():
            for child2 in child1.children():
                self._sea_blocks[child2.tags].append(child2)
    
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            seablock = make_seablock(in_planes=self.in_planes, planes=planes, stride=stride, input_dim=self.input_dim)
            # seablock = make_simple_seablock(in_planes=self.in_planes, planes=planes, stride=stride, input_dim=self.input_dim)
            layers.append(seablock)
            
            curr_shape = seablock.forward().shape
            self.input_dim = curr_shape[-1]
            self.in_planes = curr_shape[1]
        
        return nn.Sequential(*layers)
        
    def forward(self, x=None):
        if x is None:
            x = self._input_data
        
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.core_layers(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)

