#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:30:56 2019

@author: pkorus
"""
import re
import os
import imageio
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

image_id = 13

df = df[df['image_id'] == image_id]
df_j = df_j[df_j['image_id'] == image_id]
df_j2 = df_j2[df_j2['image_id'] == image_id]

#df_j = df[df['codec'] == 'jpeg']
df_d = df[df['codec'] != 'jpeg']
df_d = df_d.sort_values('rank')

df_d['layers'] = df_d['codec'].apply(lambda x : '-'.join(x.split('/')[-1].split('r:')[0].split('-')[1:]))
df_d['quantization'] = df_d['codec'].apply(lambda x : re.findall('.*r:(.*)-S.', x)[0])
df_d['codebook'] = df_d['codec'].apply(lambda x : re.findall('.*r:(.*)-Q.', x)[0])
df_d['rank'] = (1 - df_d['ssim'])**2 + (0 - df_d['bpp'])**2
df_d['label'] = df_d[['layers', 'quantization', 'ssim', 'bpp']].apply(lambda x : '{0}/{1} - {3:.2f}bpp ssim:{2:.2f} '.format(*x.values), axis=1)

fig = plt.figure()

axes = fig.gca()

sns.lineplot(data=df_j, x='bpp', y=metric, ax=axes)
sns.lineplot(data=df_j2, x='bpp', y=metric, ax=axes)
sns.scatterplot(data=df_d, x='bpp', y=metric, hue='layers', style='quantization', size='quality',
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
