
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
from morph_layers import *
from helpers import set_seeds, to_numpy

torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Helpers

def test_morph(model, X, y, morph, morph_args, test_step=False):
    orig_scores = model(X)
    morph(model, morph_args)
    new_scores = model(X)
    
    assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))
    
    if test_step:
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        loss = F.cross_entropy(new_scores, y)
        loss.backward()
        opt.step()
    
    return model

# --
# Morphs

def do_add_skip(model, idx=None, use_pooling=True):
    """
        Create an additive skip connection
        
        If spatial dimensions don't agree, also adds a nn.MaxPool2d
        If number of channels don't agree, ... fails, but shouldn't ...
    """
    idx1, idx2 = idx if idx is not None else sorted(np.random.choice(model.top_layer, 2, replace=False))
    
    print("do_add_skip: %d -> %d" % (idx1, idx2), file=sys.stderr)
    
    l1_size = model(X, idx1).size()
    l2_size = model(X, idx2).size()
    
    # If same shape, simply add
    if l1_size == l2_size:
        model.add_skip(idx1, idx2, AddLayer())
        return model
    
    # If same number of channels, rescale spatial
    if use_pooling and (l1_size[1] == l2_size[1]):
        scale_1 = l1_size[-2] / l2_size[-2]
        scale_2 = l1_size[-1] / l2_size[-1]
        assert scale_1 == scale_2, "do_add_skip: use_pooling says 'input must be square!'"
        
        new_node = model.add_skip(idx1, idx2, AddLayer())
        _ = model.modify_edge(idx1, new_node, nn.MaxPool2d(scale_1))
    
     # Different number of channels
    elif use_pooling and (l1_size[1] != l2_size[1]):
        raise Exception('dimensions do not agree!')
    
    else:
        raise Exception('dimensions do not agree!')
    
    return model


def do_cat_skip(model, idx=None):
    idx1, idx2 = idx if idx is not None else sorted(np.random.choice(model.top_layer, 2, replace=False))
    
    print("do_cat_skip: %d -> %d" % (idx1, idx2), file=sys.stderr)
    
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
    
    print("do_make_wider: %d" % idx, file=sys.stderr)
    
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
    
    print("do_make_deeper: %d -> %d" % (idx1, idx2), file=sys.stderr)
    
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
# Testing basic layers

# def make_model():
#     set_seeds()
#     return SeaNet({
#         0 : (MorphConv2d(1, 32, kernel_size=3, padding=1), "data"),
#         1 : (MorphConv2d(32, 32, kernel_size=3, padding=1), 0),
#         2 : (MorphConv2d(32, 64, kernel_size=3, padding=1), 1),
#         3 : (nn.MaxPool2d(2), 2),
#         4 : (MorphFlatLinear(12544, 10), 3)
#     })


# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_add_skip, (0, 1))

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_cat_skip, (0, 1))

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_make_deeper, (0, 1))

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_make_wider, 0)


# # --
# # Testing MorphBCRLayer

# def make_model():
#     set_seeds()
#     model = SeaNet({
#         0 : (MorphConv2d(1, 32, kernel_size=3, padding=1), "data"),
#         1 : (MorphConv2d(32, 32, kernel_size=3, padding=1), 0),
#         2 : (MorphBCRLayer(32, 32, kernel_size=3, padding=1), 1),
#         3 : (nn.MaxPool2d(2), 2),
#         4 : (MorphFlatLinear(6272, 10), 3)
#     })
#     model.eval()
#     return model

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_add_skip, (0, 1))

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_cat_skip, (0, 1))

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_make_deeper, (0, 1))
# test_morph(make_model(), X, y, do_make_deeper, (1, 2))
# test_morph(make_model(), X, y, do_make_deeper, (2, 3))

# X = Variable(torch.randn(5, 1, 28, 28))
# y = Variable(torch.randperm(5))
# test_morph(make_model(), X, y, do_make_wider, 0)
# test_morph(make_model(), X, y, do_make_wider, 1)
# test_morph(make_model(), X, y, do_make_wider, 2)


# --
# Testing series of modifications

def make_model():
    set_seeds()
    model = SeaNet({
        0 : (MorphBCRLayer(1, 32, kernel_size=3, padding=1), "data"),
        1 : (nn.MaxPool2d(2), 0),
        2 : (MorphBCRLayer(32, 32, kernel_size=3, padding=1), 1),
        3 : (nn.MaxPool2d(2), 2),
        4 : (MorphBCRLayer(32, 32, kernel_size=3, padding=1), 3),
        5 : (MorphFlatLinear(1568, 10), 4),
    })
    model.eval()
    return model

np.random.seed(123)
X = Variable(torch.randn(5, 1, 28, 28))
y = Variable(torch.randperm(5))

model = make_model()
old_pred = model(X)

morph = np.random.choice([do_add_skip, do_make_deeper, do_make_wider])
print(morph)
model = morph(model)
new_pred = model(X)
np.allclose(old_pred.data.numpy(), new_pred.data.numpy())
old_pred = new_pred

print(model)
