#!/usr/bin/env python

"""
    morph_layers.py
"""

from __future__ import print_function

import sys

import torch
from torch import nn
fro mtorch.nn import MaxPool2d
from torch.autograd import Variable

from helpers import colstring

class DummyDataLayer(nn.Module):
    def __init__(self, shape):
        super(DummyDataLayer, self).__init__()
        self.shape = shape
        self.X = Variable(torch.randn(shape))
    
    def forward(self):
        return self.X
    
    def __repr__(self):
        return self.__class__.__name__ + str(self.shape)


class FlatLinear(nn.Linear):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return super(FlatLinear, self).forward(x)


class AddLayer(nn.Module):
    def __init__(self):
        super(AddLayer, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([1.0]))
    
    def forward(self, x):
        return self.alpha * x[0] + (1 - self.alpha) * x[1]
    
    def __repr__(self):
        return self.__class__.__name__ + ' + ' + str(self.alpha.data[0])


class CatLayer(nn.Module):
    def __init__(self, dim=1):
        super(CatLayer, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.cat(x, dim=self.dim)
    
    def __repr__(self):
        return self.__class__.__name__ + ' || '

# --
# Morphable layers

class MorphMixin(object):
    def __init__(self, *args, **kwargs):
        super(MorphMixin, self).__init__(*args, **kwargs)
        self.allow_morph = False
    
    def forward(self, x):
        try:
            return super(MorphMixin, self).forward(x)
        except:
            if self.allow_morph:
                self.morph_in(x)
                return super(MorphMixin, self).forward(x)
            else:
                raise


class MorphFlatLinear(MorphMixin, FlatLinear):
    def morph_in(self, x):
        new_features = x.view(x.size(0), -1).size(-1)
        
        print(colstring.green('MorphFlatLinear: %d features -> %d features' % (self.in_features, new_features)), file=sys.stderr)
        
        padding = torch.zeros((self.out_features, new_features - self.in_features))
        
        self.in_features = new_features
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=-1))


class MorphConv2d(MorphMixin, nn.Conv2d):
    def morph_in(self, x):
        new_channels = x.size(1)
        print(colstring.green('MorphConv2d: %d channels -> %d channels' % (self.in_channels, new_channels)), file=sys.stderr)
        
        # Pad weight
        padding = torch.zeros((self.out_channels, new_channels - self.in_channels) + self.kernel_size)
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=1))
        
        self.in_channels = new_channels
    
    def morph_out(self, new_channels):
        
        # Pad weight
        padding = torch.zeros((new_channels - self.out_channels, self.in_channels) + self.kernel_size)
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=0))
        
        # Pad bias
        if self.bias is not None:
            padding = torch.zeros(new_channels - self.out_channels)
            self.bias.data.set_(torch.cat([self.bias.data, padding]))
        
        self.out_channels = new_channels
    
    def set_in_channels(self, in_channels):
        self.in_channels = in_channels
    
    def to_eye(self):
        _ = self.bias.data.zero_()
        self.weight.data = torch.zeros((self.out_channels, self.in_channels) + self.kernel_size)
        mid = int((self.kernel_size[0] - 1) / 2)
        self.weight.data[:, :self.out_channels, mid, mid] = torch.eye(self.out_channels).view(-1)


class MorphBatchNorm2d(MorphMixin, nn.BatchNorm2d):
    
    """
        !! This is _not_ idempotent when mode=train
        !! Does this matter?  Not sure...
    """
    
    def __init__(self, *args, **kwargs):
        super(MorphBatchNorm2d, self).__init__(*args, **kwargs)
        self.eps = 0
    
    def morph_in(self, x):
        new_channels = x.size(1)
        print(colstring.green('MorphBatchNorm2d: %d channels -> %d channels' % (self.num_features, new_channels)), file=sys.stderr)
        self.morph_out(new_channels)
    
    def morph_out(self, new_channels):
        # Pad running_mean
        padding = torch.zeros(new_channels - self.num_features)
        self.running_mean.set_(torch.cat([self.running_mean, padding], dim=0))
        
        # Pad running_var
        padding = 1 + torch.zeros(new_channels - self.num_features)
        self.running_var.set_(torch.cat([self.running_var, padding], dim=0))
        
        if self.affine:
            # Pad weight
            padding = torch.zeros(new_channels - self.num_features)
            self.weight.data.set_(torch.cat([self.weight.data, padding], dim=0))
            
            # Pad bias
            padding = torch.zeros(new_channels - self.num_features)
            self.bias.data.set_(torch.cat([self.bias.data, padding]))
            
        self.num_features = new_channels
    
    def to_eye(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.bias.data.zero_()
            self.weight.data.zero_()
            self.weight.data += 1


class MorphBCRLayer(MorphMixin, nn.Sequential):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(MorphBCRLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = MorphConv2d(in_channels, out_channels, **kwargs)
        self.bn = MorphBatchNorm2d(out_channels)
        
        self.add_module('conv', self.conv)
        self.add_module('bn', self.bn)
        self.add_module('relu', nn.ReLU())
    
    def morph_in(self, x):
        self.conv.morph_in(x)
        self.bn.morph_in(self.conv(x))
    
    def morph_out(self, new_channels):
        self.conv.morph_out(new_channels)
        self.bn.morph_out(new_channels)
        self.out_channels = new_channels
    
    def set_in_channels(self, in_channels):
        self.in_channels = in_channels
        self.conv.set_in_channels(in_channels)
    
    def to_eye(self):
        self.conv.to_eye()
        self.bn.to_eye()
    
    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.conv.padding != (0,) * len(self.conv.padding):
            s += ', padding={padding}'
        if self.conv.dilation != (1,) * len(self.conv.dilation):
            s += ', dilation={dilation}'
        if self.conv.output_padding != (0,) * len(self.conv.output_padding):
            s += ', output_padding={output_padding}'
        if self.conv.groups != 1:
            s += ', groups={groups}'
        if self.conv.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.conv.__dict__)