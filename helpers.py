#!/usr/bin/env python

"""
    helpers.py
"""

import numpy as np

import torch
from torch.autograd import Variable

def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def set_seeds(seed=123):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available:
        _ = torch.cuda.manual_seed(seed)

class colstring(object):
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    
    @staticmethod
    def blue(msg):
        return colstring.BLUE + msg + colstring.ENDC
        
    @staticmethod
    def green(msg):
        return colstring.GREEN + msg + colstring.ENDC
        
    @staticmethod
    def yellow(msg):
        return colstring.YELLOW + msg + colstring.ENDC
        
    @staticmethod
    def red(msg):
        return colstring.RED + msg + colstring.ENDC