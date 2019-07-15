#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:19:20 2019

@author: pkorus
"""
import os
import numpy as np
import pandas as pd
from helpers import plotting, loading
from compression import ratedistortion, afi
from training import compression

from pathlib import Path

# %% Binary representations

# plots = [('dcn.csv', {'quantization': 'soft-codebook-1bpf', 'entropy_reg': 100}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
# plots = [('dcn.csv', {'quantization': 'soft-8bpf', 'entropy_reg': 1}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

plots = [('dcn-binary.csv', {}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((15, 8))
ratedistortion.plot_curve(plots, axes[0], title='DCN with binary repr.', images=[])
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], title='Example', images=[im])

# %% Performance for M-ary representations

codebooks = [1, 2, 3, 4, 5] # bpfs of latent representations
images = [0, 11, 13, 30, 36]

n_images = len(images)

fig, axes = plotting.sub((n_images + 1)*len(codebooks), ncols=n_images+1)
fig.set_size_inches((5 * (n_images+1), 4 * len(codebooks)))

for j, bpf in enumerate(codebooks):

    plots = [('dcn-m-ary.csv', {'quantization': 'soft-codebook-{:d}bpf'.format(bpf)}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

    ratedistortion.plot_curve(plots, axes[j * (n_images+1)], title='{}-bit repr.'.format(bpf), images=[], plot='ensemble')
    for i, im in enumerate(images):
        ratedistortion.plot_curve(plots, axes[j * (n_images+1) + i + 1], title='Example {}'.format(im), images=[im])

# %% Entropy-regularization
# I. Fix codebook and see impact of regularization and #features

latent_bpf = 5

plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf)}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
# plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-3bpf', 'entropy_reg': 250}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]
# images = []

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((18, 10))
ratedistortion.plot_curve(plots, axes[0], title='{}-bpf codebook w. var reg/#features'.format(latent_bpf), images=[], plot='ensemble')
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], title='Example {}'.format(im), images=[im])

# %% Entropy-regularization
# II. Fix codebook and regularization

latent_bpf = 5

# plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-3bpf'}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf), 'entropy_reg': 250}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((15, 8))
ratedistortion.plot_curve(plots, axes[0], title='{}-bpf repr.'.format(latent_bpf), images=[], plot='ensemble')
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], title='Example {}'.format(im), images=[im])

# %% Tabularized SSIM for various settings

# %% Load sample data

dataset = './data/clic256/'
images = [0, 11, 13, 30, 36]

# Discover test files
files, _ = loading.discover_files(dataset, n_images=-1, v_images=0)
batch_x = loading.load_images(files, dataset, load='y')
batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

plotting.imsc(batch_x, titles='')

# %% Show latent representations

latent_bpf = 3

dirname = './data/raw/dcn/twitter_ent10k/TwitterDCN-8192D/'

models = ['16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+1.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+10.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+50.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+100.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+250.00'.format(latent_bpf)
          ]

dcn = afi.restore_model(os.path.join(dirname, models[0]), patch_size=256)
codebook = dcn.get_codebook()

for im in images[3:4]:
    batch_z = dcn.compress(batch_x[im:im+1])
    entropies = []
    lengths_pf = []
    for n in range(batch_z.shape[-1]):
        entropies.append(utils.entropy(batch_z[:, :, :, n], codebook))
        lengths_pf.append(np.nan)

    batch_f = np.expand_dims(np.moveaxis(batch_z[:,:,:,:].squeeze(), 2, 0), axis=3)
    fig = plotting.imsc(batch_f, figwidth=20, ncols=8,
                        titles=['{}: H={:.2f}'.format(x, e) for x, e in enumerate(entropies)])

# %% Show latent distributions
# I. No entropy regularization

# dirname = './data/raw/dcn/twitter_ent10k/TwitterDCN-8192D/'
dirname = './data/raw/dcn/twitter_10k/TwitterDCN-8192D/'

models = ['16x16x32-r:identity-Q-8.0bpf-S-',
          '16x16x32-r:soft-Q-8.0bpf-S-',
          '16x16x32-r:soft-codebook-Q-1.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-2.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-3.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-4.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-5.0bpf-S+'
          ]

# dirname = './data/raw/dcn/twitter_ent10k/TwitterDCN-8192D/16x16x32-r:soft-codebook-Q-5.0bpf-S+-H+250.00'

fig, axes = plotting.sub(len(models), ncols=4)
fig.set_size_inches((len(models)*4, 3*2))

for i, model in enumerate(models):
    dcn = afi.restore_model(os.path.join(dirname, model), patch_size=256)
    batch_z = dcn.compress(batch_x)
    fig = compression.visualize_distribution(dcn, batch_x, ax=axes[i])

# %% Show latent distributions
# II With entropy regularization

latent_bpf = 3

dirname = './data/raw/dcn/twitter_ent10k/TwitterDCN-8192D/'

models = ['16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+1.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+10.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+50.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+100.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+250.00'.format(latent_bpf)
          ]

fig, axes = plotting.sub(len(models), ncols=len(models))
fig.set_size_inches((len(models)*4, 3))

for i, model in enumerate(models):
    dcn = afi.restore_model(os.path.join(dirname, model), patch_size=256)
    batch_z = dcn.compress(batch_x)
    fig = compression.visualize_distribution(dcn, batch_x, ax=axes[i])

# %% Compare global vs. per-layer entropy coding

models = []

model_directory = './data/raw/dcn/twitter_ent10k/'
models = [str(mp.parent.parent) for mp in list(Path(model_directory).glob('**/progress.json'))]

# %%

df = pd.DataFrame(columns=['model', 'image_id', 'entropy', 'entropy_min', 'entropy_max', 'global', 'layered'])

for model in models:

    dcn = afi.restore_model(model, patch_size=256)
    codebook = dcn.get_codebook()

    for id in range(batch_x.shape[0]):

        # Global compression
        coded_fse = afi.global_compress(dcn, batch_x[id:id+1])
    
        # Per-layer compression
        layer_coded_fse = afi.afi_compress(dcn, batch_x[id:id+1])
    
        # Estimate entropy
        batch_z = dcn.compress(batch_x[id:id+1])
        entropy = utils.entropy(batch_z, codebook)
    
        n_layers = batch_z.shape[-1]
        entropies = [utils.entropy(batch_z[:, :, :, n], codebook) for n in range(n_layers)]
    
        df = df.append({'model': dcn.model_code.split('/')[-1],
                   'image_id': id,
                   'entropy': entropy,
                   'entropy_min': min(entropies),
                   'entropy_max': max(entropies),
                   'global': len(coded_fse),
                   'layered': len(layer_coded_fse)
                }, ignore_index=True)

df = df.infer_objects()
df['ratio'] = df['layered'] / df['global']

print(df.groupby('model').mean().to_string())