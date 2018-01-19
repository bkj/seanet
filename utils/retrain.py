#!/usr/bin/env python

"""
    retrain.py
"""


from __future__ import print_function, division

import os
import sys
import copy
import argparse
import numpy as np
from time import time

sys.path.append('..')
from seanet import SeaNet
from trainer import make_dataloaders, train
from lr import LRSchedule
from helpers import reset_parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()


# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    
    model = SeaNet.load(args.inpath)
    model = reset_parameters(model)
    
    dataloaders = make_dataloaders(train_size=1.0)
    
    model, performance = train(model, dataloaders, epochs=args.epochs, gpu_id=1, verbose=True)
    
    model.save('./retrained')
    json.dump(performance, open('./retrained-performance.json', 'w'))
