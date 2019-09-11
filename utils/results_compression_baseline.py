#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:19:20 2019

@author: pkorus
"""
import sys
sys.path.append('..')

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

sns.set('paper', font_scale=2, style="darkgrid")
sns.set_context("paper")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

rc('axes', titlesize=14)
rc('axes', labelsize=14)
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('legend', fontsize=10)
rc('figure', titlesize=14)

from test_dcn import match_jpeg
from helpers import plotting, loading, utils
from compression import ratedistortion, afi
from training import compression

from misc import get_sample_images

dataset = '../data/clic512'

# %% Entropy-regularization
# II. Fix codebook and regularization

latent_bpf = 5

plots = OrderedDict()
plots['jpeg'] = ('jpeg.csv', {})
plots['jpeg2000'] = ('jpeg2000.csv', {})
plots['bpg'] = ('bpg.csv', {})
plots['dcn'] = ('dcn-entropy.csv', {
                  'quantization': 'soft-codebook-{}bpf'.format(latent_bpf),
                  'entropy_reg': 250
                })

dataset = '../data/kodak512/'
images = get_sample_images(dataset)

fig, axes = plt.subplots(ncols=len(images)+1, nrows=1, sharey=True)
fig.set_size_inches((20, 3))

ratedistortion.plot_curve(plots, axes[0], dataset, title=os.path.split(dataset.strip('/'))[-1], images=[], plot='ensemble')
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], dataset,
                              title='Example {}'.format(im),
                              images=[im], plot='ensemble',
                              add_legend=False, marker_legend=False)

fig.savefig('fig_dcn_tradeoff_{}.pdf'.format(os.path.split(dataset.strip('/'))[-1]),
            bbox_inches='tight')

# %% Load sample data

dataset = '../data/kodak512/'
images = get_sample_images(dataset)

# Discover test files
files, _ = loading.discover_files(dataset, n_images=-1, v_images=0)
batch_x = loading.load_images(files, dataset, load='y')
batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

fig = plotting.imsc(batch_x[images], titles='')
fig.tight_layout()

fig.savefig('fig_samples_{}.pdf'.format(os.path.split(dataset.strip('/'))[-1]),
            bbox_inches='tight',
            dpi=72)

# %% Show latent representations

latent_bpf = 5
image_id = 2
model_id = 4

dirname = '../data/raw/dcn/entropy/TwitterDCN-4096D/'

models = ['16x16x16-r:soft-codebook-Q-{:.1f}bpf-S+-H+1.00'.format(latent_bpf),
          '16x16x16-r:soft-codebook-Q-{:.1f}bpf-S+-H+10.00'.format(latent_bpf),
          '16x16x16-r:soft-codebook-Q-{:.1f}bpf-S+-H+50.00'.format(latent_bpf),
          '16x16x16-r:soft-codebook-Q-{:.1f}bpf-S+-H+100.00'.format(latent_bpf),
          '16x16x16-r:soft-codebook-Q-{:.1f}bpf-S+-H+250.00'.format(latent_bpf)
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
    fig = plotting.imsc(batch_f, figwidth=10, ncols=4,
                        titles=['Channel {}: H={:.2f}'.format(x, e) for x, e in enumerate(entropies)])

fig.savefig('fig_latent_model_{}_image_{}.pdf'.format(model_id, images[image_id]), bbox_inches='tight')

# %% Show latent distributions
# I. No entropy regularization

dirname = '../data/raw/dcn/m-ary/TwitterDCN-8192D/'

models = [
          # '16x16x32-r:identity-Q-8.0bpf-S-',
           '16x16x32-r:soft-Q-8.0bpf-S-',
          '16x16x32-r:soft-codebook-Q-1.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-2.0bpf-S+',
           '16x16x32-r:soft-codebook-Q-3.0bpf-S+',
          # '16x16x32-r:soft-codebook-Q-4.0bpf-S+',
          '16x16x32-r:soft-codebook-Q-5.0bpf-S+'
]

titles = [
        # 'real-valued',
        'integer',
        'binary',
        '2 bpf',
        # '3 bits per feature',
        '4 bpf',
        '5 bpf'
]

fig, axes = plotting.sub(len(models), ncols=5)
fig.set_size_inches((len(models)*4, 3*1))

for i, model in enumerate(models):
    dcn = afi.restore_model(os.path.join(dirname, model), patch_size=256)
    batch_z = dcn.compress(batch_x)
    fig = compression.visualize_distribution(dcn, batch_x, ax=axes[i], title=titles[i]+' :')

fig.savefig('fig_latent_dist_{}.pdf'.format('m-ary'), bbox_inches='tight')

# %% Show latent distributions
# II With entropy regularization

latent_bpf = 5

dirname = '../data/raw/dcn/entropy/TwitterDCN-8192D/'

models = ['16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+1.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+10.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+50.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+100.00'.format(latent_bpf),
          '16x16x32-r:soft-codebook-Q-{:.1f}bpf-S+-H+250.00'.format(latent_bpf)
          ]

fig, axes = plotting.sub(len(models), ncols=len(models))
fig.set_size_inches((len(models)*5, 3))

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

image_id = 0
model = '4k'

dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
fig = match_jpeg(dcn, batch_x[images[image_id]:images[image_id]+1], match='bpp')

fig.savefig('fig_jpeg_bpp_match_model_{}_image_{}.pdf'.format(model, images[image_id]), bbox_inches='tight')

# %% Show matching images

from compression import jpeg_helpers
from skimage.measure import compare_ssim

image_id = 1
model = '4k'

dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
sample_x = batch_x[images[image_id]:images[image_id]+1]

# Compress using DCN and get number of bytes
sample_y, bytes_dcn = afi.dcn_simulate_compression(dcn, sample_x)

ssim_dcn = compare_ssim(sample_x.squeeze(), sample_y.squeeze(), multichannel=True, data_range=1)
bpp_dcn = 8 * bytes_dcn / np.prod(sample_x.shape[1:-1])

try:
    q1 = jpeg_helpers.match_quality(sample_x.squeeze(), ssim_dcn, match='ssim')
except:
    q1 = 95 if ssim_dcn > 0.8 else 10

try:
    q2 = jpeg_helpers.match_quality(sample_x.squeeze(), bpp_dcn, match='bpp')
except:
    q2 = 95 if bpp_dcn > 4 else 1

# Compress using JPEG Q1
sample_q1, bytes_jpeg = jpeg_helpers.compress_batch(sample_x[0], q1, effective=True)
ssim_q1 = compare_ssim(sample_x.squeeze(), sample_q1.squeeze(), multichannel=True, data_range=1)
bpp_q1 = 8 * bytes_jpeg / np.prod(sample_x.shape[1:-1])

# Compress using JPEG Q1
sample_q2, bytes_jpeg = jpeg_helpers.compress_batch(sample_x[0], q2, effective=True)
ssim_q2 = compare_ssim(sample_x.squeeze(), sample_q2.squeeze(), multichannel=True, data_range=1)
bpp_q2 = 8 * bytes_jpeg / np.prod(sample_x.shape[1:-1])

crop_size = max([64, batch_x.shape[1] // 4])
fig = plotting.imsc(
        tuple(utils.crop_middle(x) for x in (sample_x, sample_y, sample_q1, sample_q2)),
        [
            'Crop from original ({0}$\\times${0})'.format(crop_size),
            'DCN $\\rightarrow$ ssim:{:.2f} bpp:{:.2f}'.format(ssim_dcn, bpp_dcn),
            'JPEG Q={} $\\rightarrow$ ssim:{:.2f} bpp:{:.2f}'.format(q1, ssim_q1, bpp_q1),
            'JPEG Q={} $\\rightarrow$ ssim:{:.2f} bpp:{:.2f}'.format(q2, ssim_q2, bpp_q2)
        ],
        ncols=2, figwidth=8
)
fig.tight_layout()

fig.savefig('fig_jpeg_match_{}_{}_image_{}.pdf'.format(model, os.path.split(dataset.strip('/'))[-1], images[image_id]), bbox_inches='tight')