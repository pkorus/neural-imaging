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
import matplotlib.pyplot as plt

sns.set('paper', font_scale=1, style="ticks")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)
# sns.set_context("paper")

from test_dcn import match_jpeg
from helpers import plotting, loading, utils
from compression import ratedistortion, afi
from training import compression

from collections import OrderedDict

dataset = '../data/clic512'

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

dataset = '../data/raw512'

latent_bpf = 5
dcn_model = '16k'
lcs = [1.0, 0.1, 0.01, 0.005]

fig = plt.figure(figsize=(5, 4))
axes = fig.gca()

plots = OrderedDict()
plots['jpg'] = ('jpeg.csv', {})
plots['jpeg2k'] = ('jpeg2000.csv', {})

ratedistortion.plot_curve(plots, axes, dataset, title='{}-bpf repr.'.format(latent_bpf), images=[], plot='averages')

for dcn_model in ['4k', '8k', '16k']:
    plots = OrderedDict()
    
    plots['dcn (b)'] = ('dcn-forensics.csv', {'model_dir': '{}-basic/'.format(dcn_model)})
    # plots['dcn (o)'] = ('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf), 'entropy_reg': 250})
    for lc in lcs:
        plots['dcn ({:.3f})'.format(lc)] = ('dcn-forensics.csv', {'model_dir': '{}-{:.4f}/'.format(dcn_model, lc)})

    ratedistortion.plot_curve(plots, axes, dataset,
                              title='{}-bpf repr.'.format(latent_bpf), images=[],
                              plot='averages', add_legend=dcn_model=='4k',
                              baseline_count=0)

if 'raw' in dataset:
    axes.set_xlim([0.2, 0.90])
    axes.set_ylim([0.875, 0.95])
else:
    axes.set_xlim([0.1, 1.55])
    axes.set_ylim([0.8, 1.00])
axes.grid(True, linestyle=':')

fig.savefig('dcn_forensics_tradeoff_{}.pdf'.format(dataset.split('/')[-1]), bbox_inches='tight')

# %% Compare rate-distortion profiles

dataset = '../data/clic512'
plot_types = ['averages', 'ensemble']
# plot_types = ['averages']

latent_bpf = 5
lcs = [1.0, 0.01, 0.005]
# lcs = [0.01]
images = []

plots = OrderedDict()
plots['jpg'] = ('jpeg.csv', {})
plots['jpeg2k'] = ('jpeg2000.csv', {})
plots['dcn (b)'] = ('dcn-forensics.csv', {'model_dir': '.*basic'.format(dcn_model)})
for lc in lcs:
    plots['dcn ({:.3f})'.format(lc)] = ('dcn-forensics.csv', {'model_dir': '{}{:.4f}'.format('.*', lc)})


fig, axes = plt.subplots(1, len(plot_types), sharey=True, sharex=True)

if not hasattr(axes, '__iter__'):
    axes = [axes]

for plot_type, ax in zip(plot_types, axes):

    ratedistortion.plot_curve(plots, ax, dataset,
                              title='{}-bpf repr.'.format(latent_bpf), images=images,
                              plot=plot_type, add_legend=True, marker_legend=False,
                              baseline_count=2,
                              dump_df=False)

    ax.set_xlim([0.1, 1.55])
    # axes.set_ylim([0.8, 1.00])
    ax.grid(True, linestyle=':')

fig.set_size_inches((5 * len(plot_types), 4))

# %% Load sample data

dataset = '../data/raw512/'
images = [0, 11, 13, 30, 36]

# Discover test files
files, _ = loading.discover_files(dataset, n_images=-1, v_images=0)
batch_x = loading.load_images(files, dataset, load='y')
batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

fig = plotting.imsc(batch_x, titles='')

# %% Compare many DCN models

model_directory = '../data/raw/dcn/forensics/'
models = [str(mp.parent.parent) for mp in list(Path(model_directory).glob('**/progress.json'))]
models = sorted(models)

print(models)

image_id = 3

fig, axes = plotting.sub(6 * len(models), ncols=6)

for model_id, model in enumerate(models):

    dcn = afi.restore_model(model, patch_size=batch_x.shape[1])
    match_jpeg(dcn, batch_x[images[image_id]:images[image_id]+1], axes[model_id*6:(model_id+1)*6])
    
    axes[model_id*6].set_ylabel(os.path.relpath(model, model_directory))

# fig.savefig('fig_compare_dcn_models_image_{}.pdf'.format(images[image_id]), bbox_inches='tight')

# %% Compare outputs of the basic compression with optimized ones

def nm(x):
    x = np.abs(x)
    return (x - x.min()) / (x.max() - x.min())


# Define the distribution channel
models = OrderedDict()
models[0.001]   = '../data/raw/dcn/forensics/8k-0.0010' # 95% accuracy
models[0.005]   = '../data/raw/dcn/forensics/8k-0.0050' # 89% accuracy
models[0.010]   = '../data/raw/dcn/forensics/8k-0.0100' # 85% accuracy
models[0.050]   = '../data/raw/dcn/forensics/8k-0.0500' # 72% accuracy
models[0.100]   = '../data/raw/dcn/forensics/8k-0.1000' # 65% accuracy
models[1.000]   = '../data/raw/dcn/forensics/8k-1.0000' # 62% accuracy
models[5.000]   = '../data/raw/dcn/forensics/8k-5.0000' # 62% accuracy
models['basic'] = '../data/raw/dcn/forensics/8k-basic'  # 62% accuracy

outputs = OrderedDict()
stats = OrderedDict()

image_id = 32 # 28 for clic

for model in models.keys():
    dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
    outputs[model], stats[model] = afi.dcn_compress_n_stats(dcn, batch_x[image_id:image_id+1])

print('# {}'.format(files[image_id]))
for model in models.keys():
    print('{:>10} : ssim = {:.3f} @ {:.3f} bpp'.format(model, stats[model]['ssim'][0], stats[model]['bpp'][0]))

fig = plotting.imsc(list(outputs.values()),
                    ['{} : ssim = {:.3f} @ {:.3f} bpp'.format(x, stats[x]['ssim'][0], stats[x]['bpp'][0]) for x in models.keys()],
                    figwidth=24)
fig.savefig('debug.pdf', bbox_inches='tight')

# %%

fig = plotting.imsc([nm(outputs[k] - outputs['basic']) for k in outputs.keys()],
                    ['{}'.format(x) for x in models.keys()], figwidth=24)

fig.savefig('diff.pdf', bbox_inches='tight')

# %% Images

from diff_nip import compare_images_ab_ref

use_pretrained_ref = True

if use_pretrained_ref:
    reference = outputs['basic']
else:
    reference = batch_x[image_id:image_id+1]

fig = compare_images_ab_ref(reference, outputs[1.000], outputs[0.010],
                            labels=['Pre-trained DCN' if use_pretrained_ref else 'Original image', '$\lambda=1.000$', '$\lambda=0.010$'])

axes_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('diff_dcn.pdf', bbox_inches='tight', dpi=np.ceil(batch_x.shape[1] / axes_bbox.height))

# %% Per-image SSIM & bpp detrioration stats

dataset = '../data/raw512/'
df = pd.read_csv(os.path.join(dataset, 'dcn-forensics.csv'), index_col=False)

ssim_costs = []
bpp_costs = []

transfer_model = '8k-1.0000/'

print('# {}'.format(dataset))
for filename in df['filename'].unique():
    dfc = df.loc[df['filename'] == filename]
    basic_ssim = dfc.loc[dfc['model_dir'] == '8k-basic/', 'ssim'].values[0]
    basic_bpp = dfc.loc[dfc['model_dir'] == '8k-basic/', 'bpp'].values[0]
    transfer_ssim = dfc.loc[dfc['model_dir'] == transfer_model, 'ssim'].values[0]
    transfer_bpp = dfc.loc[dfc['model_dir'] == transfer_model, 'bpp'].values[0]
    ssim_costs.append(transfer_ssim - basic_ssim)
    bpp_costs.append(transfer_bpp - basic_bpp)
    print('{:2d} {:>35} -> ssim: {:.3f} bpp: {:.3f}'.format(dfc['image_id'].values[0], filename, ssim_costs[-1], bpp_costs[-1]))
print('{:2} {:>35} -> ssim: {:.3f} bpp: {:.3f}'.format('Σ', '-', np.mean(ssim_costs), np.mean(bpp_costs)))

# %%

from models import lpips

distance = lpips.lpips(outputs['basic'], batch_x[image_id:image_id+1])

print(distance)
