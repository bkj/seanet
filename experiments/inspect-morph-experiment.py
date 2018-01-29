import json
import pandas as pd
import numpy as np
from rsub import *
from matplotlib import pyplot as plt

df = pd.DataFrame(map(json.loads, open('./morph-experiment.jl')))
del df['val']

df = df[(df.run_id >= 0) & (df.run_id < df.run_id.max())]
df['model_id'] = df.run_id.astype(str) + '-' + df.scratch.astype(int).astype(str)


sel = np.array(((df.epoch == 39) & (df.scratch)) | ((df.epoch == 19) & (~df.scratch)))
fin = df[sel].reset_index(drop=True)

keep = set(fin.run_id[fin.scratch]).intersection(fin.run_id[~fin.scratch])
fin = fin[fin.run_id.isin(keep)]

scratch = fin[fin.scratch].sort_values('run_id').reset_index(drop=True)
tuned = fin[~fin.scratch].sort_values('run_id').reset_index(drop=True)
assert (scratch.run_id == tuned.run_id).all()

(scratch.test > tuned.test).mean()


_ = plt.hist(scratch.test, 50, alpha=0.5)
_ = plt.hist(tuned.test, 50, alpha=0.5)
show_plot()


_ = plt.scatter(scratch.test, tuned.test)
show_plot()