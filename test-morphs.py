
#!/usr/bin/env python

"""
    test-morphs.py
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
import morph_ops as mo
import morph_layers as mm
from helpers import set_seeds, to_numpy, colstring

torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Helpers

def test_morph(model, morph, morph_args, test_step=False):
    orig_scores = model()
    morph(model, morph_args)
    new_scores = model()
    
    assert np.allclose(to_numpy(orig_scores), to_numpy(new_scores))
    
    if test_step:
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        loss = F.cross_entropy(new_scores, Variable(torch.randn(new_scores.size(0))))
        loss.backward()
        opt.step()
    
    return model

# --
# Testing basic layers

def make_model():
    set_seeds()
    model = SeaNet({
        1 : (mm.MorphConv2d(1, 32, kernel_size=3, padding=1), 0),
        2 : (mm.MorphConv2d(32, 32, kernel_size=3, padding=1), 1),
        3 : (mm.MorphConv2d(32, 64, kernel_size=3, padding=1), 2),
        4 : (nn.MaxPool2d(2), 3),
        5 : (mm.MorphFlatLinear(12544, 10), 4)
    })
    model.eval()
    return model


test_morph(make_model(), mo.do_add_skip, (1, 2))

test_morph(make_model(), mo.do_cat_skip, (1, 2))

test_morph(make_model(), mo.do_make_deeper, (1, 2))

test_morph(make_model(), mo.do_make_wider, 1)

# --
# Testing MorphBCRLayer

def make_model():
    set_seeds()
    model = SeaNet({
        1 : (mm.MorphConv2d(1, 32, kernel_size=3, padding=1), 0),
        2 : (mm.MorphConv2d(32, 32, kernel_size=3, padding=1), 1),
        3 : (mm.MorphBCRLayer(32, 32, kernel_size=3, padding=1), 2),
        4 : (nn.MaxPool2d(2), 3),
        5 : (mm.MorphFlatLinear(6272, 10), 4)
    })
    model.eval()
    return model

test_morph(make_model(), mo.do_add_skip, (1, 2))

test_morph(make_model(), mo.do_cat_skip, (1, 2))

test_morph(make_model(), mo.do_make_deeper, (1, 2))
test_morph(make_model(), mo.do_make_deeper, (2, 3))
test_morph(make_model(), mo.do_make_deeper, (3, 4))

test_morph(make_model(), mo.do_make_wider, 1)
test_morph(make_model(), mo.do_make_wider, 2)
test_morph(make_model(), mo.do_make_wider, 3)


# --
# Testing series of modifications

def make_model():
    set_seeds(456)
    model = SeaNet({
        1 : (mm.MorphBCRLayer(1, 32, kernel_size=3, padding=1), 0),
        2 : (nn.MaxPool2d(2), 1),
        3 : (mm.MorphBCRLayer(32, 32, kernel_size=3, padding=1), 2),
        4 : (nn.MaxPool2d(2), 3),
        5 : (mm.MorphBCRLayer(32, 32, kernel_size=3, padding=1), 4),
        6 : (mm.MorphFlatLinear(1568, 10), 5),
    })
    model.eval()
    return model

model = make_model()
orig_pred = model()
old_pred = orig_pred

morph = np.random.choice([mo.do_add_skip, mo.do_cat_skip, mo.do_make_deeper, mo.do_make_wider])
backup = copy.deepcopy(model)
model = morph(model)
new_pred = model()
assert np.allclose(old_pred.data.numpy(), new_pred.data.numpy())
assert np.allclose(orig_pred.data.numpy(), new_pred.data.numpy())
old_pred = new_pred
model.pprint()

# --

print('passed')