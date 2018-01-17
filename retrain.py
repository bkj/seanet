


from __future__ import print_function, division

import os
import sys
import copy
import argparse
import numpy as np
from time import time

from seanet import SeaNet
from trainer import make_dataloaders, train
from lr import LRSchedule
# --
# Helpers


def reset_parameters(layer):
    try:
        _ = layer.reset_parameters()
    except:
        try:
            for child in layer.children():
                reset_parameters(child)
        except:
            print(layer)


# --
# Run

model_path = 'results/models/f410fc9dd28df54712b0f32f5e0330bf-lEJmmzbY-20180110_180900'
model = SeaNet.load(model_path)

dataloaders = make_dataloaders(train_size=1.0)
model, performance = train(model, dataloaders, epochs=50, gpu_id=1, verbose=True)