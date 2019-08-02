#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:19:20 2019

@author: pkorus
"""
import sys
sys.path.append('..')

import os
import numpy as np
import pandas as pd
from pathlib import Path

import seaborn as sns
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
sns.set()
sns.set_context("paper")

from test_dcn import match_jpeg
from helpers import plotting, loading, utils
from compression import ratedistortion, afi
from training import compression

dataset = './data/clic512'

# %% Binary representations

dataset = './data/clic512'

# plots = [('dcn.csv', {'quantization': 'soft-codebook-1bpf', 'entropy_reg': 100}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
# plots = [('dcn.csv', {'quantization': 'soft-8bpf', 'entropy_reg': 1}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

plots = [('dcn-binary.csv', {}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((15, 8))
ratedistortion.plot_curve(plots, axes[0], dataset, title='DCN with binary repr.', images=[])
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], dataset, title='Example', images=[im])

# %% Performance for M-ary representations

codebooks = [1, 2, 3, 4, 5] # bpfs of latent representations
images = [0, 11, 13, 30, 36]

n_images = len(images)

fig, axes = plotting.sub((n_images + 1)*len(codebooks), ncols=n_images+1)
fig.set_size_inches((5 * (n_images+1), 4 * len(codebooks)))

for j, bpf in enumerate(codebooks):

    plots = [('dcn-m-ary.csv', {'quantization': 'soft-codebook-{:d}bpf'.format(bpf)}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

    ratedistortion.plot_curve(plots, axes[j * (n_images+1)], dataset, title='{}-bit repr.'.format(bpf), images=[], plot='ensemble')
    for i, im in enumerate(images):
        ratedistortion.plot_curve(plots, axes[j * (n_images+1) + i + 1], dataset, title='Example {}'.format(im), images=[im])

# %% Entropy-regularization
# I. Fix codebook and see impact of regularization and #features

latent_bpf = 5

plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf)}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
# plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-3bpf', 'entropy_reg': 250}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]
# images = []

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((18, 10))
ratedistortion.plot_curve(plots, axes[0], dataset, title='{}-bpf codebook w. var reg/#features'.format(latent_bpf), images=[], plot='ensemble')
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], dataset, title='Example {}'.format(im), images=[im])

# %% Entropy-regularization
# II. Fix codebook and regularization

latent_bpf = 5

plots = [
        ('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf), 'entropy_reg': 250}),
        ('jpeg.csv', {}),
        ('jpeg2000.csv', {})
]

images = [0, 11, 13, 30, 36]

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((15, 8))
ratedistortion.plot_curve(plots, axes[0], dataset, title='{}-bpf repr.'.format(latent_bpf), images=[], plot='ensemble')
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], dataset, title='Example {}'.format(im), images=[im])

fig.savefig('fig_dcn_tradeoff_{}.pdf'.format('regularized'), bbox_inches='tight')

# %% Tabularized SSIM for various settings

# %% Load sample data

dataset = '../data/clic512/'
images = [0, 11, 13, 30, 36]

# Discover test files
files, _ = loading.discover_files(dataset, n_images=-1, v_images=0)
batch_x = loading.load_images(files, dataset, load='y')
batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

fig = plotting.imsc(batch_x, titles='')

# %% Show latent representations

latent_bpf = 5
image_id = 3
model_id = 4

dirname = './data/raw/dcn/entropy/TwitterDCN-8192D/'

models = ['16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+1.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+10.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+50.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+100.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+250.00'.format(latent_bpf)
          ]

dcn = afi.restore_model(os.path.join(dirname, models[model_id]), patch_size=512)
codebook = dcn.get_codebook()

for im in images[image_id:image_id+1]:
    batch_z = dcn.compress(batch_x[im:im+1])
    entropies = []
    lengths_pf = []
    for n in range(batch_z.shape[-1]):
        entropies.append(utils.entropy(batch_z[:, :, :, n], codebook))
        lengths_pf.append(np.nan)

    batch_f = np.expand_dims(np.moveaxis(batch_z[:,:,:,:].squeeze(), 2, 0), axis=3)
    fig = plotting.imsc(batch_f, figwidth=20, ncols=8,
                        titles=['Channel {}: H={:.2f}'.format(x, e) for x, e in enumerate(entropies)])

fig.savefig('fig_latent_model_{}_image_{}.pdf'.format(model_id, images[image_id]), bbox_inches='tight')

# %% Show latent distributions
# I. No entropy regularization

dirname = './data/raw/dcn/m-ary/TwitterDCN-8192D/'

models = ['16x16x32-r:identity-Q-8.0bpf-S-',
          '16x16x32-r:soft-Q-8.0bpf-S-',
          '16x16x32-r:soft-codebook-Q-1.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-2.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-3.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-4.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-5.0bpf-S+'
]

titles = [
        'real-valued',
        'integer',
        'binary',
        '2 bits per feature',
        '3 bits per feature',
        '4 bits per feature',
        '5 bits per feature'
]

fig, axes = plotting.sub(len(models), ncols=4)
fig.set_size_inches((len(models)*4, 3*2))

for i, model in enumerate(models):
    dcn = afi.restore_model(os.path.join(dirname, model), patch_size=256)
    batch_z = dcn.compress(batch_x)
    fig = compression.visualize_distribution(dcn, batch_x, ax=axes[i], title=titles[i]+' :')

fig.savefig('fig_latent_dist_{}.pdf'.format('m-ary'), bbox_inches='tight')

# %% Show latent distributions
# II With entropy regularization

latent_bpf = 5

dirname = './data/raw/dcn/entropy/TwitterDCN-8192D/'

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
    fig = compression.visualize_distribution(dcn, batch_x, ax=axes[i], title='$\lambda_H={}$ :'.format(*re.findall('H\+([0-9]+)', model)))

fig.savefig('fig_latent_dist_{}.pdf'.format('entropy'), bbox_inches='tight')

# %% Compare global vs. per-layer entropy coding

models = []

model_directory = '../data/raw/dcn/entropy/'
models = [str(mp.parent.parent) for mp in list(Path(model_directory).glob('**/progress.json'))]

models = [x for x in models if '5.0bpf' in x and 'H+250' in x]
models = {re.findall('([0-9]{1,2})[0-9]{3}D', m)[0]+'k': m for m in models}

# %%

df = pd.DataFrame(columns=['model', 'image_id', 'entropy', 'entropy_min', 'entropy_max', 'global', 'layered'])

for model in models:

    dcn = afi.restore_model(model, patch_size=batch_x.shape[1])
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

# %% Show DCN and JPEG images at a matching SSIM quality level

image_id = 0
model = '4k'

dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
fig = match_jpeg(dcn, batch_x[images[image_id]:images[image_id]+1], match='ssim')

fig.savefig('fig_jpeg_match_model_{}_image_{}.pdf'.format(model, images[image_id]), bbox_inches='tight')

# %% Show DCN and JPEG images at a matching bpp

image_id = 3
model = '4k'

dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
fig = match_jpeg(dcn, batch_x[images[image_id]:images[image_id]+1], match='bpp')

fig.savefig('fig_jpeg_bpp_match_model_{}_image_{}.pdf'.format(model, images[image_id]), bbox_inches='tight')

# %% Compare performance of forensics models

latent_bpf = 5

plots = [
        ('dcn-forensics.csv', {}),
        ('jpeg.csv', {}),
        ('jpeg2000.csv', {})
]

images = [0, 11, 13, 21, 30, 36]

fig, axes = plotting.sub(len(images), ncols=2)
fig.set_size_inches((10, 8))
# ratedistortion.plot_curve(plots, axes[0], dataset, title='{}-bpf repr.'.format(latent_bpf), images=[], plot='ensemble')

for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i], dataset, title='Example {}'.format(im), images=[im], plot='line')

# %% Compare performance of forensics models

from collections import OrderedDict

latent_bpf = 5
dcn_model = '8k'
lcs = [1.0, 0.1, 0.01, 0.001]

plots = OrderedDict()
plots['jpg'] = ('jpeg.csv', {})
plots['jpeg2k'] = ('jpeg2000.csv', {})
plots['dcn (b)'] = ('dcn-forensics.csv', {'model_dir': '{}-basic/'.format(dcn_model)})
# plots['dcn (o)'] = ('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf), 'entropy_reg': 250})
for lc in lcs:
    plots['dcn ({:.3f})'.format(lc)] = ('dcn-forensics.csv', {'model_dir': '{}-{:.4f}/'.format(dcn_model, lc)})

images = [0, 11, 13, 21, 30, 36]

fig = plt.figure()
axes = fig.gca()
# fig.set_size_inches((10, 8))
ratedistortion.plot_curve(plots, axes, dataset, title='{}-bpf repr.'.format(latent_bpf), images=[], plot='averages')

axes.set_xlim([0.5, 1])
axes.set_ylim([0.9, 0.95])

# fig.savefig('fig_dcn_tradeoff_{}.pdf'.format('regularized'), bbox_inches='tight')

# %% Compare many DCN models

model_directory = './data/raw/dcn/forensics/'
models = [str(mp.parent.parent) for mp in list(Path(model_directory).glob('**/progress.json'))]
models = sorted(models)

print(models)

image_id = 3

fig, axes = plotting.sub(6 * len(models), ncols=6)

for model_id, model in enumerate(models):

    dcn = afi.restore_model(model, patch_size=batch_x.shape[1])
    match_jpeg(dcn, batch_x[images[image_id]:images[image_id]+1], axes[model_id*6:(model_id+1)*6])
    
    axes[model_id*6].set_ylabel(os.path.relpath(model, model_directory))

fig.savefig('fig_compare_dcn_models_image_{}.pdf'.format(images[image_id]), bbox_inches='tight')
