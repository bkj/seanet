#!/usr/bin/env python

"""
    main.py
"""

from __future__ import print_function, division

import os
import sys
import copy
import json
import argparse
import numpy as np
from time import time

from seanet import SeaNet
import morph_layers as mm
from morph_ops import do_random_morph
from helpers import set_seeds, reset_parameters
from trainer import train, make_dataloaders

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default='morph-experiment')
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


# --
# Run

args = parse_args()
print(args, file=sys.stderr)

set_seeds(args.seed)

for d in ['models', 'logs']:
    try:
        d_run = os.path.join('results', args.run_name, d)
        assert not os.path.exists(d_run), '%s already exists!' % d_run
        _ = os.makedirs(d_run)
    except:
        pass


base_model = SeaNet({
    1 : (mm.MorphBCRLayer(3, 64, kernel_size=3, padding=1), 0),
    2 : (mm.MaxPool2d(2), 1),
    3 : (mm.MorphBCRLayer(64, 128, kernel_size=3, padding=1), 2),
    4 : (mm.MaxPool2d(2), 3),
    5 : (mm.MorphBCRLayer(128, 128, kernel_size=3, padding=1), 4),
    6 : (mm.MorphFlatLinear(2048 * 4, 10), 5),
}, input_shape=(3, 32, 32))


pretrain_epochs = 20
additional_epochs = 20

dataloaders = make_dataloaders(train_size=1.0)
base_model, base_performance = train(base_model, dataloaders, epochs=pretrain_epochs, verbose=True)

for p in base_performance:
    p.update({"run_id" : -1, "scratch" : True})
    print(json.dumps(p))

# --
# Compare training morphed versions from scratch

run_id = 0
while True:
    run_id += 1
    print('run_id=%d' % run_id, file=sys.stderr)
    
    new_model = do_random_morph(base_model, n=5)
    
    # --
    # Train w/ morph
    model, performance = train(new_model, dataloaders, epochs=additional_epochs, verbose=False)
    for p in performance:
        p.update({"run_id" : run_id, "scratch" : False})
        print(json.dumps(p))
    
    # --
    # Train from scratch
    
    new_model = reset_parameters(new_model)
    model, performance = train(new_model, dataloaders, epochs=pretrain_epochs + additional_epochs, verbose=False)
    for p in performance:
        p.update({"run_id" : run_id, "scratch" : True})
        print(json.dumps(p))



