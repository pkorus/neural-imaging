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

image_id = 21

df = df[df['image_id'] == image_id]
df_j = df_j[df_j['image_id'] == image_id]
df_j2 = df_j2[df_j2['image_id'] == image_id]

#df_j = df[df['codec'] == 'jpeg']
df_d = df[df['codec'] != 'jpeg']
df_d = df_d.sort_values('rank')

df_d['layers'] = df_d['codec'].apply(lambda x : '-'.join(x.split('/')[-1].split('r:')[0].split('-')[1:]))
df_d['quantization'] = df_d['codec'].apply(lambda x : re.findall('.*r:(.*)-S.', x)[0])
df_d['entropy_reg'] = df_d['codec'].apply(lambda x : float(re.findall('.*H\+([0-9\.]+)', x)[0]))
df_d['codebook'] = df_d['codec'].apply(lambda x : re.findall('.*r:(.*)-Q.', x)[0])
df_d['latent'] = df_d['codec'].apply(lambda x : re.findall('.*/(\d+x\d+x\d+).*', x)[0])
df_d['n_features'] = df_d['latent'].apply(lambda x : int(x.split('x')[-1]))


df_d['rank'] = (1 - df_d['ssim'])**2 + (0 - df_d['bpp'])**2
df_d['label'] = df_d[['layers', 'quantization', 'ssim', 'bpp']].apply(lambda x : '{0}/{1} - {3:.2f}bpp ssim:{2:.2f} '.format(*x.values), axis=1)

fig = plt.figure(figsize=(10,7))

axes = fig.gca()

sns.lineplot(data=df_j, x='bpp', y=metric, ax=axes)
sns.lineplot(data=df_j2, x='bpp', y=metric, ax=axes)
sns.scatterplot(data=df_d, x='bpp', y=metric, hue='n_features', style='quantization', size='entropy_reg',
                ax=axes, legend='full')

axes.set_xlim([0, 5])
#axes.set_ylim([0.7, 1])

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

from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a - np.exp(-b*x**c)


dirname = '../data/clic256'
metric = 'ssim'
images = [21]

target_bpps= np.linspace(min([df['bpp'].min(), df_j['bpp'].min()]), max([df['bpp'].max(), df_j['bpp'].max()]), 100)

df = pd.read_csv(os.path.join(dirname, 'dcn.csv'), index_col=False)
df_j = pd.read_csv(os.path.join(dirname, 'jpeg.csv'), index_col=False)

if len(images) > 0:
    df['selected'] = df['image_id'].apply(lambda x : x in images)
    df_j['selected'] = df_j['image_id'].apply(lambda x : x in images)
else:
    df['selected'] = True
    df_j['selected'] = True

labels = ['twitter', 'jpeg']
styles = [['g-', 'gx'], ['r-', 'r.']]

ssim_min = 1
for index, dfc in enumerate((df, df_j)):
    
    bpps = dfc.loc[dfc['selected'], 'bpp'].values
    ssims = dfc.loc[dfc['selected'], metric].values
    
    target_bpps= np.linspace(bpps.min(), bpps.max(), 25)
    
    popt, pcov = curve_fit(func, bpps, ssims, bounds=([0, 1e-7, 1e-7], [100, 20, 3]))
    plt.plot(target_bpps, func(target_bpps, *popt), styles[index][0], label=labels[index])
    # plt.plot(bpps, ssims, styles[index][1])
    plt.plot(bpps, ssims, styles[index][1], alpha=1/(sum(df['selected'])/5))
    ssim_min = min([ssim_min, func(target_bpps[0], *popt)])
    
    # sns.scatterplot(data=dfc.loc[dfc['selected']], x='bpp', y=metric, hue='n_features', style='quantization', size='entropy_reg',
    #             ax=axes, legend='full')


plt.ylim([ssim_min * 0.99, 1])
plt.legend()
plt.title('{} : {} images'.format(
        os.path.split(dirname)[-1],
        len(dfc.loc[dfc['selected'], 'image_id'].unique()),
        ))