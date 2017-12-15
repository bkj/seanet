#!/usr/bin/env python

"""
    morphs.py
"""

from __future__ import print_function

import sys
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

class MorphFlatLinear(FlatLinear):
    def __init__(self, *args, **kwargs):
        super(MorphFlatLinear, self).__init__(*args, **kwargs)
        self.allow_morph = False
    
    def _morph(self, x):
        new_features = x.view(x.size(0), -1).size(-1)
        
        print('MorphFlatLinear: %d features -> %d features' % (self.in_features, new_features), file=sys.stderr)
        
        padding = torch.zeros((self.out_features, new_features - self.in_features))
        
        self.in_features = new_features
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=-1))
    
    def forward(self, x):
        try:
            return super(MorphFlatLinear, self).forward(x)
        except:
            if self.allow_morph:
                self._morph(x)
                return super(MorphFlatLinear, self).forward(x)
            else:
                raise

class MorphConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MorphConv2d, self).__init__(*args, **kwargs)
        self.allow_morph = False
    
    def _morph(self, x):
        new_channels = x.size(1)
        
        print('MorphConv2d: %d channels -> %d channels' % (self.in_channels, new_channels), file=sys.stderr)
        
        padding = torch.zeros((
            self.out_channels,
            new_channels - self.in_channels,
            self.kernel_size[0],
            self.kernel_size[1],
        ))
        
        self.in_channels = new_channels
        self.weight.data.set_(torch.cat([self.weight.data, padding], dim=1))
        
    def forward(self, x):
        try:
            return super(MorphConv2d, self).forward(x)
        except:
            if self.allow_morph:
                self._morph(x)
                return super(MorphConv2d, self).forward(x)
            else:
                raise

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
        model = model.add_skip(idx1, idx2, AddLayer())
    else:
        print('dimensions do not agree!')
    
    return model

def do_cat_skip(model, idx=None):
    if idx is None:
        idx1, idx2 = sorted(np.random.choice(model.top_layer, 2, replace=False))
    else:
        idx1, idx2 = idx
    
    l1_size = model(X, idx1).size()
    l2_size = model(X, idx2).size()
    
    if l1_size[-2:] == l2_size[-2:]: # Check spatial dimensions
        model = model.add_skip(idx1, idx2, CatLayer())
        
        for k, (layer, inputs) in model.graph.items():
            try:
                _ = model(X, k)
            except:
                layer.allow_morph = True
                _ = model(X, k)
                layer.allow_morph = False
    else:
        print('dimensions do not agree!')
    
    return model

# --
# Setup

torch.set_default_tensor_type('torch.DoubleTensor')

def make_model():
    set_seeds()
    return SeaNet({
        0 : (MorphConv2d(1, 32, kernel_size=3, padding=1), "data"),
        1 : (MorphConv2d(32, 32, kernel_size=3, padding=1), 0),
        2 : (nn.MaxPool2d(2), 1),
        3 : (MorphFlatLinear(6272, 10), 2)
    })

X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))

# --
# Test additive skip morph
model = make_model()
orig_scores = model(X)
model = do_add_skip(model, (0, 1))
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

