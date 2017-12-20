#!/usr/bin/env python

"""
    lr.py
    
    learning rate scheduler
"""

import numpy as np

# --
# Helpers

def power_sum(base, k):
    return (base ** (k + 1) - 1) / (base - 1)


def inv_power_sum(x, base):
    return np.log(x * (base - 1) + 1) / np.log(base) - 1

# --

class LRSchedule(object):
    
    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def constant(x, lr_init=0.1, epochs=10):
        return lr_init
    
    @staticmethod
    def step(x, breaks=(150, 250)):
        if x < breaks[0]:
            return 0.1
        elif x < breaks[1]:
            return 0.01
        else:
            return 0.001
    
    @staticmethod
    def linear(x, lr_init=0.1, epochs=10):
        return lr_init * float(epochs - x) / epochs
    
    @staticmethod
    def cyclical(x, lr_init=0.1, epochs=10):
        """ Cyclical learning rate w/ annealing """
        if x < 1:
            # Start w/ small learning rate
            return 0.05
        else:
            return lr_init * (1 - x % 1) * (epochs - np.floor(x)) / epochs
    
    @staticmethod
    def sgdr(progress, period_length=50, lr_max=0.05, lr_min=0, t_mult=1):
        if t_mult > 1:
            period_id = np.floor(inv_power_sum(progress / period_length, t_mult)) + 1
            offsets = power_sum(t_mult, period_id - 1) * period_length
            period_progress = (progress - offsets) / (t_mult ** period_id * period_length)
        
        else:
            period_progress = (progress % period_length) / period_length
        
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(period_progress * np.pi))
