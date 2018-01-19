#!/usr/bin/env python

"""
    test.py
"""


from __future__ import print_function, division

import os
import sys
import copy
import argparse
import numpy as np
from time import time

from torch.autograd import Variable

sys.path.append('..')
from seanet import SeaNet
from trainer import make_dataloaders, train
from helpers import to_numpy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    parser.add_argument('--outpath', type=str)
    return parser.parse_args()


# --
# Run

if __name__ == "__main__":
    
    args = parse_args()
    print('%s -> %s' % (args.inpath, args.outpath))
    
    model = SeaNet.load(args.inpath).cuda().eval()
    
    dataloader = make_dataloaders(root='../data', train_size=1.0)['test']
    
    all_outputs = []
    for batch_idx, (data, targets) in enumerate(dataloader):
        data = Variable(data.cuda(), volatile=True)
        
        all_outputs.append(to_numpy(model(data)))
    
    all_outputs = np.vstack(all_outputs)
    np.save(args.outpath, all_outputs)