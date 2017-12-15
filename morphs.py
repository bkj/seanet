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
        
        padding = torch.zeros((self.out_channels, new_channels - self.in_channels) + self.kernel_size)
        
        self.in_channels = new_channels
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=1))
    
    def morph_out(self, new_channels):
        
        padding = torch.zeros((new_channels - self.out_channels, self.in_channels) + self.kernel_size)
        
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=0))
        
        if self.bias is not None:
            padding = torch.zeros(new_channels - self.out_channels)
            self.bias.data = torch.cat([self.bias.data, padding])
        
        self.out_channels = new_channels


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


def do_make_deeper(model, idx=None):
    if idx is None:
        edges = list(model.get_edgelist())
        idx2, idx1 = edges[np.random.choice(len(edges))]
    else:
        idx1, idx2 = idx
    
    old_layer = model.graph[idx1][0]
    
    if isinstance(old_layer, nn.Conv2d):
        # Add idempotent Conv2d layer
        new_layer = copy.deepcopy(old_layer)
        new_layer.in_channels = new_layer.out_channels
        
        _ = new_layer.bias.data.zero_()
        new_layer.weight.data = torch.zeros((new_layer.out_channels, new_layer.in_channels) + new_layer.kernel_size)
        new_layer.weight.data[:, :, 1, 1] = torch.eye(new_layer.in_channels).view(-1)
        model.modify_edge(idx1, idx2, new_layer)
        
    else:
        raise Exception('cannot make layer deeper')
    
    return model


def do_make_wider(model, idx=None):
    if idx is None:
        idx = np.random.choice(model.top_layer, 1)[0]
    
    if isinstance(model.graph[idx][0], nn.Conv2d):
        new_channels = 2 * model.graph[idx][0].out_channels
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


# --
# Setup

torch.set_default_tensor_type('torch.DoubleTensor')

def make_model():
    set_seeds()
    return SeaNet({
        0 : (MorphConv2d(1, 32, kernel_size=3, padding=1), "data"),
        1 : (MorphConv2d(32, 52, kernel_size=3, padding=1), 0),
        2 : (MorphConv2d(52, 64, kernel_size=3, padding=1), 1),
        3 : (nn.MaxPool2d(2), 2),
        4 : (MorphFlatLinear(12544, 10), 3)
    })

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))

# --
# Test additive skip morph
model = make_model()
orig_scores = model(X)
do_add_skip(model, (0, 1))
new_scores = model(X)

assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))

opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss = F.cross_entropy(new_scores, y)
loss.backward()
opt.step()

# --
# Test cat skip morph
model = make_model()
orig_scores = model(X)
model = do_cat_skip(model, (0, 1))
new_scores = model(X)

assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))

opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss = F.cross_entropy(new_scores, y)
loss.backward()
opt.step()

# --
# Test add new layer

model = make_model()
orig_scores = model(X)
model = do_make_deeper(model, (0, 1))
new_scores = model(X)

assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))

opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss = F.cross_entropy(new_scores, y)
loss.backward()
opt.step()

# --
# Test widen layer

model = make_model()
orig_scores = model(X)
model = do_make_wider(model, 0)
new_scores = model(X)

assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))

opt = torch.optim.Adam(model.parameters(), lr=0.1)
loss = F.cross_entropy(new_scores, y)
loss.backward()
opt.step()













