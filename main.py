#!/usr/bin/env python

"""
    main.py
"""

from __future__ import print_function, division

import os
import sys
import copy
import argparse
import numpy as np
from time import time

from seanet import SeaNet
import morph_layers as mm
from morph_ops import do_random_morph
from helpers import set_seeds
from trainer import train_mp

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default='regression-test-0')
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr-init', type=float, default=0.05)
    
    parser.add_argument('--num-neighbors', type=int, default=8)
    parser.add_argument('--num-morphs', type=int, default=5)
    parser.add_argument('--num-steps', type=int, default=5)
    parser.add_argument('--num-epoch-neighbors', type=int, default=17)
    parser.add_argument('--num-epoch-final', type=int, default=100)
    
    parser.add_argument('--seed', type=int, default=123)
    
    return parser.parse_args()


# --
# Run

args = parse_args()

set_seeds(args.seed)

for d in ['results/models', 'results/logs']:
    if not os.path.exists(os.path.join(d, args.run_name)):
        os.makedirs(os.path.join(d, args.run_name))

base_model = SeaNet({
    1 : (mm.MorphBCRLayer(3, 64, kernel_size=3, padding=1), 0),
    2 : (mm.MaxPool2d(2), 1),
    3 : (mm.MorphBCRLayer(64, 128, kernel_size=3, padding=1), 2),
    4 : (mm.MaxPool2d(2), 3),
    5 : (mm.MorphBCRLayer(128, 128, kernel_size=3, padding=1), 4),
    6 : (mm.MorphFlatLinear(2048 * 4, 10), 5),
}, input_shape=(3, 32, 32))

all_models = {-1 : train_mp(
    run_name=args.run_name,
    step_id=-1,
    models={0 : base_model},
    epochs=args.epochs,
    verbose=True
)}

best_model = copy.deepcopy(all_models[-1][0]['model'])

start_time = time()
for step in range(args.num_steps):
    print(('+'* 100) + " step=%d | %fs" % (step, time() - start_time), file=sys.stderr)
    
    neibs = {0 : best_model}
    while len(neibs) < args.num_neighbors:
        try:
            neibs[len(neibs)] = do_random_morph(best_model, n=args.num_morphs)
        except KeyboardInterrupt:
            raise
        except:
            pass
        
        print(('-'* 100) + " neib=" + str(len(neibs)), file=sys.stderr)
    
    # Train (in parallel)
    all_models[step] = train_mp(
        run_name=args.run_name,
        step_id=step,
        models=neibs,
        epochs=args.num_epoch_neighbors,
        verbose=False,
    )
    best_id = np.argmax([o['performance']['test_accs'][-1] for k,o in all_models[step].items()])
    best_model = copy.deepcopy(all_models[step][best_id]['model'])
    print('best_model:', file=sys.stderr)
    print(best_model, file=sys.stderr)
