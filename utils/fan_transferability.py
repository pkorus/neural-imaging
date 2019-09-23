import os
import sys
import pandas as pd
import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt
sys.path.append('..')

from helpers import coreutils

dirname = os.path.expanduser('~/Dropbox/logs')

print(dirname)
filenames = coreutils.listdir(dirname, '.*log')

df = pd.DataFrame(columns=['training data', 'compression', 'test set', 'validation accuracy', 'test accuracy'])

for filename in filenames:

    training_spec = os.path.splitext(filename)[0].split('_')

    with open(os.path.join(dirname, filename)) as f:

        for line in f.readlines():

            if line.startswith(';'):
                result_spec = line.strip().split(';')

                df = df.append({'training data': training_spec[0],
                           'compression': training_spec[1],
                           'test set': training_spec[2],
                           'validation accuracy': float(result_spec[-1]),
                           'test accuracy': float(result_spec[-2])
                           }, ignore_index=True)


print(df.to_string())

sns.set('paper', font_scale=1, style="darkgrid")
sns.set_context("paper")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
g = sns.relplot(x='validation accuracy', y='test accuracy', hue='test set',
                col='compression', row='training data', data=df, legend='full')

for i in range(2):
    for j in range(3):
        g.axes[i, j].plot([0.2, 1], [0.2, 1], 'k:')

leg = g._legend
leg.set_bbox_to_anchor([0.05, 0.95])  # coordinates of lower left of bounding box
leg._loc = 2

g.fig.set_size_inches((12, 6))
g.fig.show()

g.fig.savefig('fig_fan_validation.pdf', bbox_inches='tight')
