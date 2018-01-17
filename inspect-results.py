#!/usr/bin/env python

"""
    inspect-results.py
"""

import json
import numpy as np
import pandas as pd
from glob import glob

pd.set_option('display.width', 150)

from rsub import *
from matplotlib import pyplot as plt

x = list(map(lambda x: json.load(open(x)), glob('results/logs/run-n12-m2-s20/*')))

y = []
for xx in x:
    for epoch, x in enumerate(xx['performance']):
        x.update({
            "epoch" : epoch,
            "step_id" : xx['step_id'],
            "model_id" : xx['model_id'],
            "timestamp" : xx['timestamp'],
            "model_path" : xx['model_path'],
        })
        
        y.append(x)

df = pd.DataFrame(y).sort_values(['step_id', 'model_id', 'epoch']).reset_index(drop=True)
df = df[['model_path', 'timestamp', 'step_id', 'model_id', 'epoch', 'train', 'val', 'test']]

df = df[df.test > 0.2].reset_index(drop=True)
df['uid'] = df.step_id.astype(str) + '-' + df.model_id.astype(str)

for uid in df.uid.unique():
    sub = df[df.uid == uid]
    
    s = sub.step_id * sub.epoch.max() + np.arange(sub.shape[0])
    p = np.array(sub.test)
    _ = plt.plot(s, p, alpha=0.25)

_ = plt.ylim(0.7, 1.0)
show_plot()
