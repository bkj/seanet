#!/usr/bin/env python

"""
    morphs.py
"""

from __future__ import print_function

import sys
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from seanet import SeaNet
from helpers import set_seeds, to_numpy

# --
# Layers

class FlatLinear(nn.Linear):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return super(FlatLinear, self).forward(x)


class AddLayer(nn.Module):
    def __init__(self, alpha=1):
        super(AddLayer, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha]))
    
    def forward(self, x):
        return self.alpha * x[0] + (1 - self.alpha) * x[1]
    
    def __repr__(self):
        return self.__class__.__name__ + ' + '


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
        
        print('MorphFlatLinear: %d features -> %d features' % (self.in_features, new_features), file=sys.stderr)
        
        padding = torch.zeros((self.out_features, new_features - self.in_features))
        
        self.in_features = new_features
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=-1))


class MorphConv2d(MorphMixin, nn.Conv2d):
    def morph_in(self, x):
        new_channels = x.size(1)
        print('MorphConv2d: %d channels -> %d channels' % (self.in_channels, new_channels), file=sys.stderr)
        
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
        self.weight.data[:, :, 1, 1] = torch.eye(self.in_channels).view(-1)


class MorphBatchNorm2d(MorphMixin, nn.BatchNorm2d):
    
    """
        !! This is _not_ idempotent when mode=train
        !! Does this matter?  Not sure...
    """
    
    def morph_in(self, x):
        new_channels = x.size(1)
        print('MorphBatchNorm2d: %d channels -> %d channels' % (self.num_features, new_channels), file=sys.stderr)
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
        print('MorphBCRLayer: vv', file=sys.stderr)
        self.conv.morph_in(x)
        self.bn.morph_in(self.conv(x))
    
    def morph_out(self, new_channels):
        self.conv.morph_out(new_channels)
        self.bn.morph_out(new_channels)
    
    def set_in_channels(self, in_channels):
        self.in_channels = in_channels
        self.conv.in_channels = self.conv.in_channels
    
    def to_eye(self):
        self.conv.to_eye()
        self.bn.to_eye()


# --
# Morphs

def do_add_skip(model, idx=None):
    if idx is None:
        idx1, idx2 = sorted(np.random.choice(model.top_layer, 2, replace=False))
    else:
        idx1, idx2 = idx
    
    l1_size = model(X, idx1).size()
    l2_size = model(X, idx2).size()
    
    if l1_size == l2_size: # Check all dimensions
        model.add_skip(idx1, idx2, AddLayer())
    else:
        raise Exception('dimensions do not agree!')
    
    return model


def do_cat_skip(model, idx=None):
    if idx is None:
        idx1, idx2 = sorted(np.random.choice(model.top_layer, 2, replace=False))
    else:
        idx1, idx2 = idx
    
    l1_size = model(X, idx1).size()
    l2_size = model(X, idx2).size()
    
    if l1_size[-2:] == l2_size[-2:]: # Check spatial dimensions
        model.add_skip(idx1, idx2, CatLayer())
        
        for k, (layer, inputs) in model.graph.items():
            try:
                _ = model(X, k)
            except:
                layer.allow_morph = True
                _ = model(X, k)
                layer.allow_morph = False
    else:
        raise Exception('dimensions do not agree!')
    
    return model


def do_make_wider(model, idx=None, k=2):
    if idx is None:
        idx = np.random.choice(model.top_layer, 1)[0]
    
    old_layer = model.graph[idx][0]
    
    if isinstance(old_layer, MorphConv2d) or isinstance(old_layer, MorphBCRLayer):
        new_channels = k * old_layer.out_channels
        model.graph[idx][0].morph_out(new_channels=new_channels)
        
        for k, (layer, inputs) in model.graph.items():
            try:
                _ = model(X, k)
            except:
                layer.allow_morph = True
                _ = model(X, k)
                layer.allow_morph = False
    else:
        raise Exception('cannot make layer wider')
    
    return model


def do_make_deeper(model, idx=None):
    if idx is None:
        edges = list(model.get_edgelist())
        idx2, idx1 = edges[np.random.choice(len(edges))]
    else:
        idx1, idx2 = idx
    
    old_layer = model.graph[idx1][0]
    
    if isinstance(old_layer, MorphConv2d) or isinstance(old_layer, MorphBCRLayer):
        new_layer = copy.deepcopy(old_layer)
        new_layer.set_in_channels(new_layer.out_channels)
        new_layer.to_eye()
        model.modify_edge(idx1, idx2, new_layer)
        
    else:
        raise Exception('cannot make layer deeper')
    
    return model

# --
# Setup

torch.set_default_tensor_type('torch.DoubleTensor')

def make_model():
    set_seeds()
    return SeaNet({
        0 : (MorphConv2d(1, 32, kernel_size=3, padding=1), "data"),
        1 : (MorphConv2d(32, 32, kernel_size=3, padding=1), 0),
        2 : (MorphConv2d(32, 64, kernel_size=3, padding=1), 1),
        3 : (nn.MaxPool2d(2), 2),
        4 : (MorphFlatLinear(12544, 10), 3)
    })

def test_morph(model, X, y, morph, morph_args):
    orig_scores = model(X)
    morph(model, morph_args)
    new_scores = model(X)
    
    assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))
    
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss = F.cross_entropy(new_scores, y)
    loss.backward()
    opt.step()
    
    return model


X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_add_skip, (0, 1))

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_cat_skip, (0, 1))

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_make_deeper, (0, 1))

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_make_wider, 0)


# --
# 

def make_model():
    set_seeds()
    model = SeaNet({
        0 : (MorphConv2d(1, 32, kernel_size=3, padding=1), "data"),
        1 : (MorphConv2d(32, 32, kernel_size=3, padding=1), 0),
        2 : (MorphBCRLayer(32, 32, kernel_size=3, padding=1), 1),
        3 : (nn.MaxPool2d(2), 2),
        4 : (MorphFlatLinear(6272, 10), 3)
    })
    model.eval()
    return model

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_add_skip, (0, 1))

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_cat_skip, (0, 1))

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_make_deeper, (0, 1))
test_morph(make_model(), X, y, do_make_deeper, (1, 2))
test_morph(make_model(), X, y, do_make_deeper, (2, 3)) # This fails, because of extra `eps` division in BN

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))
test_morph(make_model(), X, y, do_make_wider, 0)
test_morph(make_model(), X, y, do_make_wider, 1)
test_morph(make_model(), X, y, do_make_wider, 2)
