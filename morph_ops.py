
#!/usr/bin/env python

"""
    morph_ops.py
"""

from __future__ import print_function

import sys
import copy
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from morph_layers import *
from helpers import  colstring

torch.set_default_tensor_type('torch.DoubleTensor')

def do_add_skip(model, idx=None, fix_channels=True, fix_spatial=True):
    idx1, idx2 = idx if idx is not None else sorted(np.random.choice(model.top_layer, 2, replace=False))
    
    print(colstring.blue("do_add_skip: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    l1_size = model(layer=idx1).size()
    l2_size = model(layer=idx2).size()
    
    if l1_size[-1] > l2_size[-1]:
        raise Exception(colstring.red('do_add_skip: (l1_size[-1] > l2_size[-1]) -- can cause downstream problems'))
    
    # Add skip
    merge_node = model.add_skip(idx1, idx2, AddLayer())
    
    # If same shape, return
    if l1_size != l2_size:
        # If different channels, add a convolutional block on skip connection
        if (l1_size[1] != l2_size[1]):
            if fix_channels:
                new_layer = MorphBCRLayer(l1_size[1], l2_size[1], kernel_size=3, padding=1)
                idx1 = model.modify_edge(idx1, merge_node, new_layer)
            else:
                raise Exception(colstring.red('do_add_skip: (l1_size[1] != l2_size[1]) and not fix_channels'))
        
        # If different spatial extent, add max pooling
        if (l1_size[-1] != l2_size[-1]):
            if fix_spatial:
                scale = l1_size[-1] / l2_size[-1]
                _ = model.modify_edge(idx1, merge_node, nn.MaxPool2d(scale))
            else:
                raise Exception(colstring.red('do_add_skip: (l1_size[-1] != l2_size[-1]) and not fix_spatial'))
    
    model.compile()
    return model


def do_cat_skip(model, idx=None, fix_spatial=True):
    idx1, idx2 = idx if idx is not None else sorted(np.random.choice(model.top_layer, 2, replace=False))
    
    print(colstring.blue("do_cat_skip: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    l1_size = model(layer=idx1).size()
    l2_size = model(layer=idx2).size()
    
    # Add skip
    merge_node = model.add_skip(idx1, idx2, CatLayer())
    
    # If different spatial extent, add max pooling
    if l1_size[-1] != l2_size[-1]:
        if l1_size[-1] > l2_size[-1]:
            scale = l1_size[-1] / l2_size[-1]
            maxpool_node = model.modify_edge(idx1, merge_node, nn.MaxPool2d(scale))
        else:
            scale = l2_size[-1] / l1_size[-1]
            maxpool_node = model.modify_edge(idx2, merge_node, nn.MaxPool2d(scale))
    
    # 1x1 conv to fix output dim
    out_size = model(layer=merge_node).size()
    new_layer = MorphConv2d(out_size[1], l2_size[1], kernel_size=1, padding=0)
    new_layer.to_eye()
    model.modify_edge(merge_node, None, new_layer)
    
    model.compile()
    return model


def do_make_wider(model, idx=None, k=2):
    if idx is None:
        idx = np.random.choice(range(1, model.top_layer), 1)[0]
    
    print(colstring.blue("do_make_wider: %d" % idx), file=sys.stderr)
    
    backup_model = copy.deepcopy(model)
    old_layer = model.graph[idx][0]
    
    if isinstance(old_layer, MorphConv2d) or isinstance(old_layer, MorphBCRLayer):
        
        # Make layer wider
        new_channels = k * old_layer.out_channels
        model.graph[idx][0].morph_out(new_channels=new_channels)
        
        # Fix downstream conflicts
        try:
            model.fix_shapes()
        except:
            # If impossible, revert changes
            model = backup_model
            return model
            raise Exception(colstring.red('do_make_wider: model.fix_shapes() has downstream conflicts'))
        
        model.compile()
        return model
        
    else:
        raise Exception(colstring.red('do_make_wider: invalid layer type -> %s' % old_layer.__class__.__name__))


def do_make_deeper(model, idx=None):
    if idx is None:
        edges = list(model.get_edgelist())
        idx2, idx1 = edges[np.random.choice(len(edges))]
    else:
        idx1, idx2 = idx
    
    print(colstring.blue("do_make_deeper: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    old_layer = model.graph[idx1][0]
    
    if isinstance(old_layer, MorphConv2d) or isinstance(old_layer, MorphBCRLayer):
        # Copy layer
        new_layer = copy.deepcopy(old_layer)
        
        # Make idempontent new layer
        new_layer.set_in_channels(new_layer.out_channels)
        new_layer.to_eye()
        
        # Add to graph
        _ = model.modify_edge(idx1, idx2, new_layer)
        
        model.compile()
        return model
        
    else:
        raise Exception(colstring.red("cannot make layer deeper -> %s" % old_layer.__class__.__name__))