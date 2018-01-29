#!/usr/bin/env python

"""
    helpers.py
"""

import os
import sys
import numpy as np
from time import time

import torch
from torch.autograd import Variable

def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def set_seeds(seed=123):
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed + 1)
    if torch.cuda.is_available:
        _ = torch.cuda.manual_seed(seed + 2)

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


last_time = time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time()  # Reset for new bar.
    
    cur_time = time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    
    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    
    msg = ''.join(L)
    sys.stderr.write(msg)
    
    if current < total-1:
        sys.stderr.write('\r')
    else:
        sys.stderr.write('\n')
    sys.stderr.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def reset_parameters(model):
    for child in model.children():
        try:
            child.reset_parameters()
        except:
            print("Cannot reset: %s" % str(child), file=sys.stderr)
    
    return model