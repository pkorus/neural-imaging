#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:30:56 2019

@author: pkorus
"""
import re
import os
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

# %%

dirname = '../data/clic256'
metric = 'ssim'

df = pd.read_csv(os.path.join(dirname, 'dcn.csv'), index_col=False)

df_j = pd.read_csv(os.path.join(dirname, 'jpeg.csv'), index_col=False)
df_j2 = pd.read_csv(os.path.join(dirname, 'jpeg2000.csv'), index_col=False)


df['rank'] = (1 - df['ssim'])**2 + (0 - df['bpp']/5)**2
df_j['rank'] = (1 - df_j['ssim'])**2 + (0 - df_j['bpp']/5)**2

image_id = 0

df = df[df['image_id'] == image_id]
df_j = df_j[df_j['image_id'] == image_id]
df_j2 = df_j2[df_j2['image_id'] == image_id]

#df_j = df[df['codec'] == 'jpeg']
df_d = df[df['codec'] != 'jpeg']
df_d = df_d.sort_values('rank')

df_d['rank'] = (1 - df_d['ssim'])**2 + (0 - df_d['bpp'])**2
df_d['label'] = df_d[['layers', 'quantization', 'ssim', 'bpp']].apply(lambda x : '{0}/{1} - {3:.2f}bpp ssim:{2:.2f} '.format(*x.values), axis=1)

fig = plt.figure(figsize=(10,7))

axes = fig.gca()

sns.lineplot(data=df_j, x='bpp', y=metric, ax=axes)
sns.lineplot(data=df_j2, x='bpp', y=metric, ax=axes)
sns.scatterplot(data=df_d, x='bpp', y=metric, hue='quantization', style='n_features', size='entropy_reg', ax=axes, legend='full')

# axes.set_xlim([1, 5])
# axes.set_ylim([0.90, 1])

# %% Get to 10

df_selected = df_d[df_d['codebook'] != 'identity']
df_selected = df_selected[0:10]

images = []

for index, row in df_selected.iterrows():
    
    image = imageio.imread(os.path.join(dirname, 
                                        os.path.splitext(row['filename'])[0], 
                                        row['codec'].replace('/', '-') + '.png'))
    
    images.append(image)
    
    print(row['codec'])
    
# %%
from helpers import plotting
    
fig = plotting.imsc(images, df_selected['label'].tolist(), figwidth=10)

# %%


# %%

fig = plt.figure()

axes = fig.gca()

bpps = df_j['bpp'].values
ssims = df_j[metric].values


weights = np.exp(-10 * np.power(bpps.reshape((-1,1)) - target_bpps.reshape((1,-1)),2)) + 1e-5
weights /= np.sum(weights, axis=0)

plt.plot(target_bpps, (ssims.reshape((1,-1)) @ weights).flatten(), 'r-')
plt.plot(bpps, ssims, 'r.', alpha=0.25)



bpps = df['bpp'].values
ssims = df[metric].values

target_bpps= np.linspace(bpps.min(), bpps.max(), 25)

weights = np.exp(-10 * np.power(bpps.reshape((-1,1)) - target_bpps.reshape((1,-1)),2)) + 1e-16
weights /= np.sum(weights, axis=0)

plt.plot(target_bpps, (ssims.reshape((1,-1)) @ weights).flatten(), 'g-')
plt.plot(bpps, ssims, 'gx', alpha=0.5)

# df['latent'] = df['codec'].apply(lambda x : re.findall('[0-9]+x[0-9]+x[0-9]+', x)[0])
# df['features'] = df['latent'].apply(lambda x : x.split('x')[-1])



# %%

df_selected = df_d[df_d['codebook'] != 'identity']
df_selected = df_selected[0:10]

images = []

for index, row in df_selected.iterrows():
    
    image = imageio.imread(os.path.join(dirname, 
                                        os.path.splitext(row['filename'])[0], 
                                        row['codec'].replace('/', '-') + '.png'))
    
    images.append(image)
    
    print(row['codec'])
    
# %%
from helpers import plotting
    
fig = plotting.imsc(images, df_selected['label'].tolist(), figwidth=10)

