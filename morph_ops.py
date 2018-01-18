
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

import morph_layers as mm
from helpers import colstring, to_numpy


class InvalidMorphException(Exception):
    pass


def do_add_skip(model, idx=None, fix_channels=True, fix_spatial=True):
    if idx is None:
        idx1, idx2 = model.random_nodes(n=2, allow_input=True)
    else:
        idx1, idx2 = idx
    
    print(colstring.blue("do_add_skip: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    # --
    # Check that morph is valid
    
    l1_size = model(layer=idx1).size()
    l2_size = model(layer=idx2).size()
    
    if l1_size[-1] < l2_size[-1]:
        print(colstring.red('do_add_skip: (l1_size[-1] > l2_size[-1]) -- can cause downstream problems'), file=sys.stderr)
        return None
    
    # Set flags
    _need_fix_channels, _need_fix_spatial = False, False
    if l1_size != l2_size:
        if l1_size != l2_size:
            if fix_channels:
                print(colstring.red('do_add_skip: !fix_channels'), file=sys.stderr)
                _need_fix_channels = True
            else:
                return None
        
        if (l1_size[-1] != l2_size[-1]):
            if fix_spatial:
                print(colstring.red('do_add_skip: !fix_spatial'), file=sys.stderr)
                _need_fix_spatial = True
            else:
                return None
    
    # --
    # Create morph function
    
    def f(model, idx1=idx1, idx2=idx2):
        l1_size = model(layer=idx1).size()
        l2_size = model(layer=idx2).size()
        
        merge_node = model.add_skip(idx1, idx2, AddLayer())
        
        if _need_fix_channels:
            new_layer = mm.MorphBCRLayer(l1_size[1], l2_size[1], kernel_size=3, padding=1)
            idx1 = model.modify_edge(idx1, merge_node, new_layer)
        
        if _need_fix_spatial:
            scale = int(l1_size[-1] / l2_size[-1])
            _ = model.modify_edge(idx1, merge_node, nn.MaxPool2d(scale))
                    
        model.compile()
        return model
    
    return f


def do_cat_skip(model, idx=None, fix_spatial=True):
    if idx is None:
        idx1, idx2 = model.random_nodes(n=2, allow_input=True)
    else:
        idx1, idx2 = idx
    
    print(colstring.blue("do_cat_skip: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    # --
    # Check that morph is valid
    
    l1_size = model(layer=idx1).size()
    l2_size = model(layer=idx2).size()
    
    if l1_size[-1] < l2_size[-1]: # !! ??
        print(colstring.red('do_add_skip: (l1_size[-1] > l2_size[-1]) -- can cause downstream problems'), file=sys.stderr)
        return None
    
    # --
    # Create morph function
    
    def f(model, idx1=idx1, idx2=idx2):
        l1_size = model(layer=idx1).size()
        l2_size = model(layer=idx2).size()
        
        # Add skip
        merge_node = model.add_skip(idx1, idx2, CatLayer())
        
        # If different spatial extent, add max pooling
        if l1_size[-1] != l2_size[-1]:
            if l1_size[-1] > l2_size[-1]:
                scale = int(l1_size[-1] / l2_size[-1])
                maxpool_node = model.modify_edge(idx1, merge_node, nn.MaxPool2d(scale))
            else:
                scale = int(l2_size[-1] / l1_size[-1])
                maxpool_node = model.modify_edge(idx2, merge_node, nn.MaxPool2d(scale))
        
        # 1x1 conv to fix output dim
        out_size = model(layer=merge_node).size()
        new_layer = mm.MorphConv2d(out_size[1], l2_size[1], kernel_size=1, padding=0)
        new_layer = new_layer.to_eye()
        model.modify_edge(merge_node, None, new_layer)
        
        model.compile()
        return model
    
    return f


def do_make_wider(model, idx=None, k=2):
    if idx is None:
        idx = model.random_nodes(n=1, allow_input=False)[0]
    
    print(colstring.blue("do_make_wider: %d" % idx), file=sys.stderr)
    
    # --
    # Check that morph is valid
    
    old_layer = model.get_node(idx)
    if not isinstance(old_layer, mm.MorphConv2d) and not isinstance(old_layer, mm.MorphBCRLayer):
        print(colstring.red('do_make_wider: invalid layer type -> %s' % old_layer.__class__.__name__), file=sys.stderr)
        return None
    
    try:
        backup_model = copy.deepcopy(model)
        new_channels = k * old_layer.out_channels
        backup_model.get_node(idx).morph_out(new_channels=new_channels)
        backup_model.fix_shapes()
    except:
        print(colstring.red('do_make_wider: model.fix_shapes() has downstream conflicts'), file=sys.stderr)
        return None
    
    # --
    # Create morph function
    
    def f(model, idx=idx):
        new_channels = k * model.get_node(idx).out_channels
        model.get_node(idx).morph_out(new_channels=new_channels)
        model.fix_shapes()
        
        model.compile()
        return model
    
    return f

def do_make_deeper(model, idx=None):
    if idx is None:
        idx2, idx1 = model.random_edge()
    else:
        idx1, idx2 = idx
    
    print(colstring.blue("do_make_deeper: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    # --
    # Check that morph is valid
    
    old_layer = model.get_node(idx1)
    if not isinstance(old_layer, mm.MorphConv2d) or not isinstance(old_layer, mm.MorphBCRLayer):
        print(colstring.red("cannot make layer deeper -> %s" % old_layer.__class__.__name__), file=sys.stderr)
        return None
    
    # --
    # Create morph function
    
    def f(model, idx1=idx1, idx2=idx2):
        new_layer = copy.deepcopy(model.get_node(idx1))
        new_layer.set_in_channels(new_layer.out_channels)
        new_layer = new_layer.to_eye()
        _ = model.modify_edge(idx1, idx2, new_layer)
        
        model.compile()
        return model
    
    return f

# >>

def do_flat_insert(model, idx=None):
    if idx is None:
        idx2, idx1 = model.random_edge()
    else:
        idx1, idx2 = idx
    
    print(colstring.blue("do_flat_insert: %d -> %d" % (idx1, idx2)), file=sys.stderr)
    
    old_layer1 = model.get_node(idx1)
    old_layer2 = model.get_node(idx2)
    
    layer_factories = [
        lambda c: mm.MorphConv2d(c, c, kernel_size=3, padding=1).to_eye(),
        lambda c: mm.MorphBatchNorm2d(c).to_eye(),
        lambda c: mm.ReLU(),
    ]
    
    layer_factory = layer_factories[np.random.choice(len(layer_factories))]
    
    def f(model, idx1=idx1, idx2=idx2, layer_factory=layer_factory):
        channels = model.forward(layer=idx1).shape[1]
        new_layer = layer_factory(channels)
        _ = model.modify_edge(idx1, idx2, new_layer)
        
        model.compile()
        return model
    
    return f


# <<

def _do_random_morph(model, morph_factories, assert_eye=True, max_attempts=10):
    new_model = copy.deepcopy(model)
    
    # --
    # Sample valid morph
    
    morph_factory = np.random.choice(morph_factories)
    morph = morph_factory(new_model)
    counter = 0
    while morph is None:
        morph_factory = np.random.choice(morph_factories)
        morph = morph_factory(new_model)
        counter += 1
        if counter > max_attempts:
            raise Exception('!! _do_random_morph: counter > max_attempts')
    
    # --
    # Apply morph
    
    new_model = morph(new_model)
    
    # --
    # Check idempotence
    
    if assert_eye:
        is_eye = np.allclose(to_numpy(new_model()), to_numpy(model()))
        if not is_eye:
            print('---- error ----', file=sys.stderr)
            print(model)
            print('**')
            print(new_model)
            raise Exception('_do_random_morph: assert_eye failed')
        
    
    return new_model


def do_random_morph(model, n=1, assert_eye=True):
    model = model.cpu().eval()
    
    for i in range(n):
        print('morph=%d' % i)
        model = _do_random_morph(
            model=model,
            morph_factories=[
                do_add_skip,
                do_cat_skip,
                do_make_deeper,
                do_make_wider,
                do_flat_insert,
            ],
            assert_eye=assert_eye,
        )
    
    return model

