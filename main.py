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
from datetime import datetime

from seanet import SeaNet
import morph_layers as mm
from morph_ops import do_random_morph
from helpers import set_seeds
from trainer import train_mp

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True)
    
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
    models={0 : base_model},
    epochs=args.epochs,
    verbose=True
)}

best_model = copy.deepcopy(all_models[-1][0]['model'])

for step in range(args.num_steps):
    print(('+'* 100) + " step=" + str(step), file=sys.stderr)
    
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
        models=neibs,
        epochs=args.num_epoch_neighbors,
        verbose=False,
    )
    best_id = np.argmax([o['performance']['test_accs'][-1] for k,o in all_models[step].items()])
    best_model = copy.deepcopy(all_models[step][best_id]['model'])
    print('best_model:')
    print(best_model)


# >>> pprint([(k,sorted([v['performance']['test_accs'][-1] for v in a.values()])) for k,a in all_models.items()])
# [(0, [0.1, 0.1, 0.1, 0.1, 0.8697, 0.8803, 0.8896, 0.8918]),
#  (1, [0.1, 0.895, 0.8989, 0.9014, 0.9021, 0.9022, 0.9032, 0.9071]),
#  (2, [0.1, 0.9118, 0.914, 0.9144, 0.9153, 0.9156, 0.9156, 0.9174]),
#  (3, [0.1, 0.8425, 0.9172, 0.9197, 0.9227, 0.9233, 0.9238, 0.9267]),
#  (4, [0.9274, 0.9292, 0.9302, 0.9306, 0.9307, 0.9308, 0.9308, 0.9314]),
#  (-1, [0.8586])]
# >>> pprint([(k,sorted([v['performance']['train_accs'][-1] for v in a.values()])) for k,a in all_models.items()])
# [(0, [0.1, 0.1, 0.1, 0.1, 0.91286, 0.91672, 0.93082, 0.93318]),
#  (1, [0.1, 0.9347, 0.9396, 0.9452, 0.94558, 0.94598, 0.94802, 0.95962]),
#  (2, [0.1, 0.97418, 0.97622, 0.97638, 0.97642, 0.97652, 0.97748, 0.97986]),
#  (3, [0.1, 0.85376, 0.97278, 0.97388, 0.98156, 0.9858, 0.98682, 0.98756]),
#  (4, [0.98702, 0.98896, 0.99012, 0.99056, 0.99058, 0.99078, 0.99138, 0.99142]),
#  (-1, [0.88328])]
