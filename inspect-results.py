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

x = list(map(lambda x: json.load(open(x)), glob('results/logs/val-0.8-v3/*')))

y = []
for i, xx in enumerate(x):
    for epoch, x in enumerate(xx['performance']):
        x.update({
            "id" : i,
            "epoch" : epoch,
            "step_id" : xx['step_id'],
            "model_id" : xx['model_id'],
            "timestamp" : xx['timestamp'],
            "model_path" : xx['model_path'],
        })
        
        y.append(x)

df = pd.DataFrame(y).sort_values('epoch').reset_index(drop=True)

len(set(df.model_path[df.train < 0.2]))
len(set(df.model_path))

df = df[df.test > 0.2].reset_index(drop=True)
df['uid'] = df.step_id.astype(str) + '-' + df.model_id.astype(str)


for uid in df.uid.unique():
    sub = df[df.uid == uid]
    
    s = np.arange(sub.shape[0])
    p = np.array(sub.val)
    _ = plt.plot(s, p, alpha=0.25)

_ = plt.ylim(0.8, 0.95)
show_plot()
