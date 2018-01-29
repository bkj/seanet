import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

sys.path.append('..')
from trainer import make_dataloaders
from helpers import to_numpy

# --
# IO

# Get metadata
log_fs = np.array(sorted(glob('../results/logs/run-n12-m2-s20/*')))
meta = list(map(lambda x: json.load(open(x)), log_fs))

meta = pd.DataFrame([{
    "model_name" : os.path.basename(x['log_path']),
    "timestamp" : x['timestamp']
} for x in meta])
meta = meta.sort_values('timestamp').reset_index(drop=True)

# Get labels
dataloader = make_dataloaders(root='../data', train_size=1.0)['test']
targets = np.hstack([to_numpy(targets) for _,targets in dataloader])

# Get predictions
fs = np.array(sorted(glob('../results/predictions/run-n12-m2-s20/*')))

all_preds = []
for f in tqdm(fs):
    all_preds.append(np.load(f))

all_preds = np.array(all_preds)

# Compute accuracy
accs = np.array([(all_preds[i].argmax(axis=1) == targets).mean() for i in range(all_preds.shape[0])])

# sel = accs > 0.8
# fs, all_preds, accs = fs[sel], all_preds[sel], accs[sel]


# --
# Do ensembles help?  And how big do they need to be?

def sim_ens(n):
    sel = np.random.choice(all_preds.shape[0], n)
    return (all_preds[sel].mean(axis=0).argmax(axis=1) == targets).mean()

s = np.arange(1, 30, 1)
res = []
for n in tqdm(s):
    res.append([sim_ens(n=n) for _ in range(30)])

res = np.column_stack(res)

for r in res:
    _ = plt.scatter(s, r, alpha=0.25, c='grey')

_ = plt.plot(s, res.mean(axis=0), c='red')
_ = plt.axhline(accs.max(), c='blue')
show_plot()

# Looks like it saturates around ~ 10 randomly chosen models

sort_by_time = np.array([list(meta.model_name).index(os.path.basename(i).split('.')[0]) for i in fs])
fs, all_preds, accs = fs[sort_by_time], all_preds[sort_by_time], accs[sort_by_time]

sel = accs > 0.8
fs, all_preds, accs = fs[sel], all_preds[sel], accs[sel]


ens_accs = [(all_preds[:i].mean(axis=0).argmax(axis=1) == targets).mean() for i in range(1, 1 + 20)]
best_acc = [accs[:i].max() for i in range(1, 1 + 20)]
_ = plt.plot(ens_accs[:20])
_ = plt.plot(best_acc[:20], c='grey')
_ = plt.plot(accs[:20], c='grey')
show_plot()




# >>>>>>>>>

all_preds = all_preds[np.argsort(accs)]
accs = accs[np.argsort(accs)]

sel = accs > 0.5
all_preds, accs = all_preds[sel], accs[sel]

_ = plt.plot([(all_preds[:i].mean(axis=0).argmax(axis=1) == targets).mean() for i in range(1, 1 + len(all_preds))])
_ = plt.plot([(all_preds[max(0, i-10):i].mean(axis=0).argmax(axis=1) == targets).mean() for i in range(1, 1 + len(all_preds))])
_ = plt.plot([(all_preds[max(0, i-5):i].mean(axis=0).argmax(axis=1) == targets).mean() for i in range(1, 1 + len(all_preds))])
_ = plt.plot(accs, c='grey')
show_plot()




# Accuracy of ensemble as a function of accuracy
ts = np.linspace(0.80, accs.max(), 20)
ens_accs = [(all_preds[accs >= t].mean(axis=0).argmax(axis=1) == targets).mean() for t in ts]

_ = plt.plot(ts, ens_accs)
_ = plt.ylim(0.9, 0.95)
show_plot()


def sim_ens(n):
    sel = np.random.choice(all_preds.shape[0], n)
    return (all_preds[sel].mean(axis=0).argmax(axis=1) == targets).mean()

n_runs = 20
res = []

s = np.arange(5, 140, 5)
for n in tqdm(s):
    res.append([sim_ens(n=n) for _ in range(n_runs)])

res = np.column_stack(res)

for r in res:
    _ = plt.scatter(s, r, alpha=0.25, c='grey')

_ = plt.plot(s, res.mean(axis=0), c='red')
_ = plt.axhline(accs.max(), c='blue')
show_plot()





