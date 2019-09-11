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
import imageio
from pathlib import Path

import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

sns.set('paper', font_scale=2, style="darkgrid")
sns.set_context("paper")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

from test_dcn import match_jpeg
from helpers import plotting, loading, utils
from compression import ratedistortion, afi
from training import compression

from misc import get_sample_images

from collections import OrderedDict

dataset = '../data/clic512'

# %% Show changes in rate distortion (averages)

dataset = '../data/clic512'

latent_bpf = 5
lcs = [1.0, 0.1, 0.01, 0.005]

fig = plt.figure(figsize=(8, 6))
axes = fig.gca()

plots = OrderedDict()
plots['jpg'] = ('jpeg.csv', {})
plots['jpeg2k'] = ('jpeg2000.csv', {})
plots['bpg'] = ('bpg.csv', {})

ratedistortion.plot_curve(plots, axes, dataset, title='{}-bpf repr.'.format(latent_bpf), images=[], plot='averages')

for dcn_model in ['4k', '8k', '16k']:
    plots = OrderedDict()
    
    plots['dcn (b)'] = ('dcn-forensics-7m.csv', {'model_dir': '{}-basic/'.format(dcn_model)})
    # plots['dcn (o)'] = ('dcn-entropy.csv', {'quantization': 'soft-codebook-{}bpf'.format(latent_bpf), 'entropy_reg': 250})
    for lc in lcs:
        plots['dcn ({:.3f})'.format(lc)] = ('dcn-forensics-7m.csv', {'model_dir': '{}-{:.4f}/'.format(dcn_model, lc)})

    ratedistortion.plot_curve(plots, axes, dataset,
                              title='{}-bpf repr.'.format(latent_bpf), images=[],
                              plot='averages', add_legend=dcn_model=='4k',
                              baseline_count=0)

if 'raw' in dataset:
    axes.set_xlim([0.25, 1.15])
    axes.set_ylim([0.875, 0.97])
else:
    axes.set_xlim([0.1, 1.95])
    axes.set_ylim([0.8, 1.00])
axes.grid(True, linestyle=':')

fig.savefig('dcn_forensics_tradeoff_{}.pdf'.format(dataset.split('/')[-1]), bbox_inches='tight')

# %% Show rate distortion curves

dataset = '../data/raw512'
plot_types = ['averages', 'ensemble']
# plot_types = ['averages']

latent_bpf = 5
lcs = [1.0, 0.01, 0.005]
# lcs = [0.01]
images = []

plots = OrderedDict()
plots['jpg'] = ('jpeg.csv', {})
plots['jpeg2k'] = ('jpeg2000.csv', {})
plots['bpg'] = ('bpg.csv', {})

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
                              baseline_count=3,
                              dump_df=False)

    ax.set_xlim([0.1, 1.55])
    # axes.set_ylim([0.8, 1.00])
    ax.grid(True, linestyle=':')

fig.set_size_inches((5 * len(plot_types), 4))

# %% Load sample data

dataset = '../data/clic512/'
images = get_sample_images(dataset)

# Discover test files
files, _ = loading.discover_files(dataset, n_images=-1, v_images=0)
batch_x = loading.load_images(files, dataset, load='y')
batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

fig = plotting.imsc(batch_x[images], titles='')

# %% Detailed breakdown of the impact on a specific image

# Accuracy information will be collected from:
#  - ../results/summary-dcn-all.csv
#
# > ./results.py ./data/raw/m/dcn+ --df ./results df
# I typically shorten the scenario name, e.g.:
# 8k/D90/DNet/fixed-nip/lc-0.0010 -> 8k/0.0010

from diff_nip import compare_images_ab_ref, fft_log_norm, nm
from helpers.utils import dct_mask
import scipy as sp

dcn_model = '8k'
compact = True

# Define the distribution channel
models = OrderedDict()
models['basic'] = '../data/raw/dcn/forensics-7m/{}-basic'.format(dcn_model)
models[1.000]   = '../data/raw/dcn/forensics-7m/{}-1.0000'.format(dcn_model)
models[0.050]   = '../data/raw/dcn/forensics-7m/{}-0.0500'.format(dcn_model)
models[0.010]   = '../data/raw/dcn/forensics-7m/{}-0.0100'.format(dcn_model)
models[0.001]   = '../data/raw/dcn/forensics-7m/{}-0.0010'.format(dcn_model)

dcns = OrderedDict()
for model in models.keys():
    dcns[model] = afi.restore_model(models[model], patch_size=batch_x.shape[1])

for image_id in get_sample_images(dataset) + [10]:

    out_filename = 'debug/{}/{}_{:02d}_{}.pdf'.format(
            dataset.strip('/').split('/')[-1],
            dcn_model, image_id, 'c' if compact else 'f')

    accuracies = OrderedDict()
    outputs = OrderedDict()
    stats = OrderedDict()
    
    df_acc = pd.read_csv(os.path.join('../results/', 'summary-dcn-all-7m.csv'), index_col=False)
    
    # Get compressed images for all selected models
    for model in models.keys():
    
        if type(model) is not str:
                model_label = '{:.4f}'.format(model)
        else:
            model_label = model
    
        # dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
        outputs[model], stats[model] = afi.dcn_compress_n_stats(dcns[model], batch_x[image_id:image_id+1])
        accuracies[model] = df_acc.loc[df_acc['scenario'] == '{}/{}'.format(dcn_model, model_label), 'accuracy'].mean()
    
    all_labels = ['Original']
    all_images = [batch_x[image_id:image_id+1]]
    for key, value in outputs.items():

        if type(key) is not str:
                model_label = '$\\lambda_c$ = {:.3f}'.format(key)
        else:
            model_label = key

        all_images.append(value)
        all_labels.append('{} $\\rightarrow$ H: {:.2f}, ssim: {:.2f}'.format(model_label,
                          stats[key]['entropy'][0],
                          stats[key]['ssim'][0],
                          )) # accuracies[key]
    
    fft_images = []
    # Get DFTs and frequency profiles
    for image in all_images:
        fft_images.append(fft_log_norm(image, 0.1))
    
    # FFTs of diffs
    fft_diffs = []
    # Get DFTs and frequency profiles
    for image in all_images:
        fft_diffs.append(fft_log_norm((image - all_images[0]), 1))
    
    bands = np.linspace(0, 1, 64)
    energies = []
    
    for image in all_images:
        dct = np.abs(sp.fftpack.dct(sp.fftpack.dct(np.mean(image.squeeze(), axis=2).T).T))
    
        energy = np.zeros((len(bands), 1))
    
        # Get frequency energies
        for b in range(len(bands)):
            m = dct_mask(dct.shape[1], bands[b], 1e-2)
            energy[b] = np.mean(dct * m)
    
        energies.append(energy)
    
    # % Plot the breakdown
    
    n_plots = 2 if compact else 6
    
    fig, axes = plt.subplots(n_plots, len(all_images))
    fig.set_size_inches((3.75 * len(all_images), 3.75 * n_plots))
    
    # The images
    for index, image in enumerate(all_images):
        plotting.quickshow(image, all_labels[index], axes=axes[0, index])
        if index == 0:
            axes[0, index].set_ylabel('Original/compressed')
    
    if not compact:
        for index, image in enumerate(all_images):
            if index > 0:
                plotting.quickshow((np.abs(image - all_images[0]) * 10).clip(0, 1),
                                          '', axes=axes[1, index])
            else:
                axes[1, index].set_axis_off()
        
            if index == 1:
                axes[1, index].set_ylabel('Difference wrt. original')
        
        for index, image in enumerate(all_images):
            if index > 1:
                plotting.quickshow((np.abs(image - all_images[1]) * 10).clip(0, 1),
                                           '', axes=axes[2, index])
            else:
                axes[2, index].set_axis_off()
        
            if index == 2:
                axes[2, index].set_ylabel('Difference wrt. basic DCN')
    
    # The FFTs
    for index, image in enumerate(fft_images):
        plotting.quickshow(image, '', axes=axes[1 if compact else 3, index])
        if index == 0:
            axes[1 if compact else 3, index].set_ylabel('FFTs')
    
    if not compact:
        # for index, image in enumerate(fft_images):
        #     if index > 1:
        #         plotting.quickshow(nm(np.abs(image - fft_images[1])),
        #                                    '', axes=axes[4, index])
        #     else:
        #         axes[4, index].set_axis_off()
        
        #     if index == 2:
        #         axes[4, index].set_ylabel('FFT diff wrt. basic DCN')
    
        for index, image in enumerate(fft_diffs):
            if index > 0:
                plotting.quickshow(image, '', axes=axes[4, index])
            else:
                axes[4, index].set_axis_off()
        
            if index == 1:
                axes[4, index].set_ylabel('FFT of diff wrt. originals')
    
        # The frequency breakdown
        for index, energy in enumerate(energies):
            axes[5, index].semilogy(bands, energy)
            if index == 0:
                axes[5, index].set_ylabel('Sub-band energy')
            else:
                axes[5, index].set_yticklabels([])
            axes[5, index].set_xlabel('Relative frequency')
            axes[5, index].set_ylim([np.min(energies)/2, 2*np.max(energies)])
            axes[5, index].grid(True, linestyle=':')

    fig.subplots_adjust(wspace=0, hspace=0.05)
    os.makedirs('debug/{}/{}/'.format(dataset.strip('/').split('/')[-1], image_id), exist_ok=True)
    fig.savefig(out_filename, bbox_inches='tight', dpi=200)
    
    print('Written to: {}'.format(out_filename))

# %% Plot aggregated spectrum statistics

dcn_model = '16k'

# Define the distribution channel
models = OrderedDict()
models['basic'] = '../data/raw/dcn/forensics-7m/{}-basic'.format(dcn_model)
models[1.000]   = '../data/raw/dcn/forensics-7m/{}-1.0000'.format(dcn_model)
models[0.050]   = '../data/raw/dcn/forensics-7m/{}-0.0500'.format(dcn_model)
models[0.010]   = '../data/raw/dcn/forensics-7m/{}-0.0100'.format(dcn_model)
models[0.005]   = '../data/raw/dcn/forensics-7m/{}-0.0050'.format(dcn_model)
models[0.001]   = '../data/raw/dcn/forensics-7m/{}-0.0010'.format(dcn_model)

dcns = OrderedDict()
for model in models.keys():
    dcns[model] = afi.restore_model(models[model], patch_size=batch_x.shape[1])

outputs = OrderedDict()

for image_id in range(batch_x.shape[0]):

    outputs[image_id] = OrderedDict()

    # Get compressed images for all selected models
    for model in models.keys():
        outputs[image_id][model] = dcns[model].process(batch_x[image_id:image_id+1])

# %% Construct labels ------------------------------------------------------------

model_mapping = {'4k': '16-C', '8k': '32-C', '16k': '64-C'}

accuracies = OrderedDict()
df_acc = pd.read_csv(os.path.join('../results/', 'summary-dcn-all-7m.csv'), index_col=False)

# Get compressed images for all selected models
for model in models.keys():

    if type(model) is not str:
            model_label = '{:.4f}'.format(model)
    else:
        model_label = model

    accuracies[model] = df_acc.loc[df_acc['scenario'] == '{}/{}'.format(dcn_model, model_label), 'accuracy'].mean()

all_labels = []
for key, value in accuracies.items():

    if type(key) is not str:
            model_label = '$\\lambda_c$ = {:.3f}'.format(key)
    else:
        model_label = key

    all_labels.append('{} $\\rightarrow$ acc: {:.2f}'.format(model_label, accuracies[key]))

# The actual plotting ---------------------------------------------------------

from misc import spectrum

spectrums = []
for mid, model in enumerate(models.keys()):
    y = np.zeros(batch_x.shape)
    
    for image_id in range(batch_x.shape[0]):
        y[image_id] = spectrum((outputs[image_id][model]).squeeze()) / spectrum((batch_x[image_id]).squeeze())

    yv = y.mean(axis=(0))
    yv = sp.ndimage.median_filter(yv, 7)
    yv = np.log(yv)
    yv = yv / np.abs(yv).max()
    yv = (yv + 1) / 2
    spectrums.append(yv)

    print(model, yv.min(), yv.max())

fig = plotting.imsc(spectrums, all_labels, ncols=len(models))
fig.set_size_inches((3.25 * len(models), 3.25))
fig.subplots_adjust(wspace=0, hspace=0)
# fig.tight_layout()
out_filename = 'fig_spectral_{}.pdf'.format(dcn_model)
fig.savefig(out_filename, bbox_inches='tight',  dpi=100)

# %% JPEG Output

# TODO Temporary code to collect aggregated JPEG output
models = OrderedDict()
for v in [95, 85, 75, 50, 30]:
    models[v] = None

outputs = OrderedDict()
all_labels = []

for image_id in range(batch_x.shape[0]):

    outputs[image_id] = OrderedDict()

    # Get compressed images for all selected models
    for model in models.keys():
        outputs[image_id][model], _ = jpeg_helpers.compress_batch(batch_x[image_id:image_id+1], model)
        all_labels.append('JPEG Q={}'.format(model))

# %% Detailed breakdown of the impact on a specific image (JPEG)

# Accuracy information will be collected from:
#  - ../results/summary-dcn-all.csv
#
# > ./results.py ./data/raw/m/dcn+ --df ./results df
# I typically shorten the scenario name, e.g.:
# 8k/D90/DNet/fixed-nip/lc-0.0010 -> 8k/0.0010

from diff_nip import compare_images_ab_ref, fft_log_norm, nm
from helpers.utils import dct_mask
import scipy as sp

dcn_model = '8k'
# Worst images for clic: 1, 28, 33, 36
image_id = 0 #images[0] # 32 # 28 for clic
compact = False

out_filename = 'debug/{}/{}_{:02d}_{}.pdf'.format(
        dataset.strip('/').split('/')[-1],
        'jpeg', image_id, 'c' if compact else 'f')


# accuracies = OrderedDict()
outputs = OrderedDict()
# stats = OrderedDict()

# df_acc = pd.read_csv(os.path.join('../results/', 'summary-dcn-all.csv'), index_col=False)

# Get compressed images for all selected models
for model in [95, 90, 75, 50, 30]:

    if type(model) is not str:
            model_label = '{:.4f}'.format(model)
    else:
        model_label = model

    # dcn = afi.restore_model(models[model], patch_size=batch_x.shape[1])
    outputs[model], _ = jpeg_helpers.compress_batch(batch_x[image_id:image_id+1], model)
    # accuracies[model] = df_acc.loc[df_acc['scenario'] == '{}/{}'.format(dcn_model, model_label), 'accuracy'].mean()

all_labels = ['Original']
all_images = [batch_x[image_id:image_id+1]]
for key, value in outputs.items():
    all_images.append(value)
    all_labels.append('JPEG: {}'.format(key))

fft_images = []
# Get DFTs and frequency profiles
for image in all_images:
    fft_images.append(fft_log_norm(image, 0.1))

# FFTs of diffs
fft_diffs = []
# Get DFTs and frequency profiles
for image in all_images:
    fft_diffs.append(fft_log_norm((image - all_images[0]), 1))

bands = np.linspace(0, 1, 64)
energies = []

for image in all_images:
    dct = np.abs(sp.fftpack.dct(sp.fftpack.dct(np.mean(image.squeeze(), axis=2).T).T))

    energy = np.zeros((len(bands), 1))

    # Get frequency energies
    for b in range(len(bands)):
        m = dct_mask(dct.shape[1], bands[b], 1e-2)
        energy[b] = np.mean(dct * m)

    energies.append(energy)

# % Plot the breakdown

n_plots = 2 if compact else 6

fig, axes = plt.subplots(n_plots, len(all_images))
fig.set_size_inches((3 * len(all_images), 3 * n_plots))

# The images
for index, image in enumerate(all_images):
    plotting.quickshow(image, all_labels[index], axes=axes[0, index])
    if index == 0:
        axes[0, index].set_ylabel('Original/compressed')

if not compact:
    for index, image in enumerate(all_images):
        if index > 0:
            plotting.quickshow((np.abs(image - all_images[0]) * 10).clip(0, 1),
                                      '', axes=axes[1, index])
        else:
            axes[1, index].set_axis_off()
    
        if index == 1:
            axes[1, index].set_ylabel('Difference wrt. original')
    
    for index, image in enumerate(all_images):
        if index > 1:
            plotting.quickshow((np.abs(image - all_images[1]) * 10).clip(0, 1),
                                       '', axes=axes[2, index])
        else:
            axes[2, index].set_axis_off()
    
        if index == 2:
            axes[2, index].set_ylabel('Difference wrt. basic DCN')

# The FFTs
for index, image in enumerate(fft_images):
    plotting.quickshow(image, '', axes=axes[1 if compact else 3, index])
    if index == 0:
        axes[1 if compact else 3, index].set_ylabel('FFTs')

if not compact:
    # for index, image in enumerate(fft_images):
    #     if index > 1:
    #         plotting.quickshow(nm(np.abs(image - fft_images[1])),
    #                                    '', axes=axes[4, index])
    #     else:
    #         axes[4, index].set_axis_off()
    
    #     if index == 2:
    #         axes[4, index].set_ylabel('FFT diff wrt. basic DCN')

    for index, image in enumerate(fft_diffs):
        if index > 0:
            plotting.quickshow(image, '', axes=axes[4, index])
        else:
            axes[4, index].set_axis_off()
    
        if index == 1:
            axes[4, index].set_ylabel('FFT of diff wrt. originals')

    # The frequency breakdown
    for index, energy in enumerate(energies):
        axes[5, index].semilogy(bands, energy)
        if index == 0:
            axes[5, index].set_ylabel('Sub-band energy')
        else:
            axes[5, index].set_yticklabels([])
        axes[5, index].set_xlabel('Relative frequency')
        axes[5, index].set_ylim([np.min(energies)/2, 2*np.max(energies)])
        axes[5, index].grid(True, linestyle=':')



fig.tight_layout()
os.makedirs('debug/{}/{}/'.format(dataset.strip('/').split('/')[-1], image_id), exist_ok=True)
fig.savefig(out_filename, bbox_inches='tight', dpi=200)

print('Written to: {}'.format(out_filename))


# %% Per-image SSIM & bpp detrioration stats

dataset = '../data/raw512/'
df = pd.read_csv(os.path.join(dataset, 'dcn-forensics-7m.csv'), index_col=False)

# ssim_costs = []
# bpp_costs = []

transfer_model = '8k-1.0000/'
basic_model = '8k-basic/'

df_s = pd.DataFrame(columns=['filename', 'ssim', 'bpp'])

print('# {}'.format(dataset))
for filename in df['filename'].unique():
    dfc = df.loc[df['filename'] == filename]
    basic_ssim = dfc.loc[dfc['model_dir'] == basic_model, 'ssim'].values[0]
    basic_bpp = dfc.loc[dfc['model_dir'] == basic_model, 'bpp'].values[0]
    transfer_ssim = dfc.loc[dfc['model_dir'] == transfer_model, 'ssim'].values[0]
    transfer_bpp = dfc.loc[dfc['model_dir'] == transfer_model, 'bpp'].values[0]
    df_s = df_s.append({'filename': filename, 'ssim': transfer_ssim - basic_ssim, 'bpp': transfer_bpp - basic_bpp}, ignore_index=True)

print(df_s.sort_values('ssim'))
print('{:2} -> ssim: {:.3f} bpp: {:.3f}'.format('Σ', *df_s.mean()))

fig = plt.figure(figsize=(8,4))
ax = fig.gca()
ax.plot([0, 0], [0, 30], ':k')
ax.hist(df_s['ssim'], bins=np.linspace(-0.1, 0.1, 30))
ax.set_xlabel('SSIM deterioration in channel model')
ax.set_ylabel('Frequency')
ax.set_title(os.path.split(dataset.strip('/'))[-1])
ax.set_ylim([0, 17])
ax.set_xlim([-0.08, 0.08])

fig.savefig('fig_ssim_diff_{}.pdf'.format(os.path.split(dataset.strip('/'))[-1]), dpi=100)

# %% Pool compression quality and classification accuracy

# Accuracy information will be collected from:
#  - ../data/{}/dcn-forensics.csv       # image quality stats (ssim, bpp)
#  - ../data/{}/jpeg.csv                # image quality stats (ssim, bpp)
#  - ../results/summary-dcn-all.csv     # accuracy
#  - ../results/summary-jpeg.csv        # accuracy
#
# > ./results.py ./data/raw/m/dcn+ --df ./results df
# I typically shorten the scenario name, e.g.:
# 8k/D90/DNet/fixed-nip/lc-0.0010 -> 8k/0.0010
#
# > ./results.py ./data/raw/m/jpeg --df ./results df

dataset = '../data/clic512/'
dcn_models = ['4k', '8k', '16k']
lambdas = [0.001, 0.005, 0.010, 0.050, 0.100, 1.000, 'basic']

models = OrderedDict()

# --- DCNs --------------------------------------------------------------------
df = pd.read_csv(os.path.join(dataset, 'dcn-forensics-7m.csv'), index_col=False)
df_acc = pd.read_csv(os.path.join('../results/', 'summary-dcn-all-7m.csv'), index_col=False)

df_o = pd.DataFrame(columns=['dcn_model', 'lambda', 'bpp', 'ssim', 'accuracy'])

for dcn_model in dcn_models:
    for lbda in lambdas:

        if type(lbda) is not str:
            lbda = '{:.4f}'.format(lbda)
        models['{}-{}'.format(dcn_model, lbda)] = '../data/raw/dcn/forensics/{}-{}'.format(dcn_model, lbda)

        # Find the average bpp and ssim
        bpp = df.loc[df['model_dir'] == '{}-{}/'.format(dcn_model, lbda), 'bpp'].mean()
        ssim = df.loc[df['model_dir'] == '{}-{}/'.format(dcn_model, lbda), 'ssim'].mean()
        accuracy = df_acc.loc[df_acc['scenario'] == '{}/{}'.format(dcn_model, lbda), 'accuracy'].mean()

        df_o = df_o.append({
                'dcn_model': '{}/{}'.format(dcn_model, lbda),
                'lambda': lbda,
                'bpp': bpp,
                'ssim': ssim,
                'accuracy': accuracy
        }, ignore_index=True)

# --- JPEG --------------------------------------------------------------------
df = pd.read_csv(os.path.join(dataset, 'jpeg.csv'), index_col=False)
df_acc = pd.read_csv(os.path.join('../results/', 'summary-jpeg-7m.csv'), index_col=False)

df_j = pd.DataFrame(columns=['quality', 'bpp', 'ssim', 'accuracy'])

for quality in sorted(df['quality'].unique()):

    # Find the average bpp and ssim
    bpp = df.loc[df['quality'] == quality, 'bpp'].mean()
    ssim = df.loc[df['quality'] == quality, 'ssim'].mean()
    accuracy = df_acc.loc[df_acc['scenario'] == quality, 'accuracy'].mean()

    df_j = df_j.append({
            'quality': quality,
            'bpp': bpp,
            'ssim': ssim,
            'accuracy': accuracy
    }, ignore_index=True)


# %% Draw the summary plot

sns.set('paper', font_scale=2, style="darkgrid")
sns.set_context("paper")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

fig = plt.figure()
ax = fig.gca()

x_axis = 'bpp'
y_axis = 'accuracy'

if 'raw' in dataset:
    x_limits = [0.25, 0.95]
else:
    x_limits = [0.35, 1.75]

min_ssim = 0.85
max_ssim = max([df_j['ssim'].max(), df_o['ssim'].max()])

# Add line connecting basic DCN models ----------------------------------------
g = sns.lineplot(x=x_axis, y=y_axis, data=df_o.loc[df_o['lambda'] == 'basic'])

# Add points for basic DCN models
g = sns.scatterplot(x=x_axis, y=y_axis, size='ssim', hue='ssim', edgecolor='black',
                sizes=(20, 100), hue_norm=(min_ssim, max_ssim), size_norm=(min_ssim, max_ssim),
                legend=False, marker='o',
                s=50, alpha=0.7, data=df_o.loc[df_o['lambda'] == 'basic'])

# Add points for transferred DCN models
g = sns.scatterplot(x=x_axis, y=y_axis, hue='ssim', size='ssim', edgecolor='gray',
                sizes=(20, 100), size_norm=(min_ssim, max_ssim), hue_norm=(min_ssim, max_ssim),
                legend='brief', marker='o',
                s=50, alpha=0.7, data=df_o.loc[df_o['lambda'] != 'basic'])

# Add annotations next to transferred DCN models
for pid in range(0, len(df_o)):
    if df_o.bpp[pid] < x_limits[-1]:

        # Show annotations as: lambda [(ssim diff)]
        # The ssim diff is shown if abs() > 0.005 and color depends on the
        # sign of the difference.
        label_color = 'black'
        dcn = df_o.dcn_model[pid].split('/')[0]
        lbda = df_o.dcn_model[pid].split('/')[-1]
        if lbda != 'basic':
            # Find base SSIM
            base_ssim = df_o.loc[df_o['dcn_model'] == '{}/basic'.format(dcn), 'ssim'].values[0]
            ssim_diff = df_o.ssim[pid] - base_ssim
            if np.abs(ssim_diff) > 0.005:
                label = '{} ({:+.3f})'.format(lbda.strip('0'), ssim_diff)
            else:
                label = lbda.strip('0')

            if ssim_diff < 0:
                label_color = '#990000'
            if ssim_diff > 0:
                label_color = '#004400'
        else:
            label = '{} ({:.2f})'.format(lbda, base_ssim)

        # Draw the actual label
        ax.text(df_o.bpp[pid] + 0.015, df_o.accuracy[pid], label,
                horizontalalignment='left', size=7, color=label_color)

# Add lines & points for JPEG -------------------------------------------------
g = sns.lineplot(x=x_axis, y=y_axis, data=df_j, dashes=True)

g = sns.scatterplot(x=x_axis, y=y_axis, hue='ssim', size='ssim', edgecolor='gray',
                sizes=(20, 100), size_norm=(min_ssim, max_ssim), hue_norm=(min_ssim, max_ssim),
                legend=False, marker='D', s=50, alpha=0.7, data=df_j)

for pid in range(0, len(df_j)):
    if df_j.bpp[pid] < x_limits[-1] and int(df_j.quality[pid]) % 10 == 0:
        ax.text(df_j.bpp[pid] + 0.02, df_j.accuracy[pid], int(df_j.quality[pid]),
             horizontalalignment='left', size=7, color='black')

# Final touches ---------------------------------------------------------------
for line in g.lines:
    line.set_linestyle(':')

# Sometimes seaborn messes up legend entries - number of decimal places explodes
for t in ax.legend_.texts:
    try:
        t.set_text('{:.2f}'.format(float(t.get_text())))
    except:
        pass

g.figure.set_size_inches((8, 6))
ax.grid(True, linestyle=':')
ax.set_xlim(x_limits)
# ax.set_title('{} dataset'.format(dataset.strip('/').split('/')[-1]))
ax.set_xlabel('Effective bpp')
ax.set_ylabel('FAN accuracy')
# ax.legend(loc='lower right', fancybox=True, framealpha=0.5)

fig.tight_layout()
# fig.savefig('fig_{}_vs_{}_{}.pdf'.format(y_axis, x_axis, dataset.strip('/').split('/')[-1], image_id, dcn_model),
            # dpi=100)

# %% Accuracy vs SSIM

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

fig = plt.figure()
ax = fig.gca()

axis_labels = {'bpp': 'Effective bpp', 'accuracy': 'FAN accuracy', 'ssim': 'SSIM'}

x_axis = 'bpp'
z_axis = 'accuracy'
y_axis = 'ssim'

if 'raw' in dataset:
    x_limits = [0.225, 1.225]
else:
    x_limits = [0.4, 2.1]

if y_axis == 'ssim':
    x_shift, y_shift = 0, 0.01
    y_limits = [0.825, 0.972]

if y_axis == 'accuracy':
    x_shift, y_shift = 0.015, 0.0
    y_limits = [0.25, 1.01]


min_z = min([df_j[z_axis].min(), df_o[z_axis].min()])
max_z = max([df_j[z_axis].max(), df_o[z_axis].max()])

# Add line connecting basic DCN models ----------------------------------------
g = sns.lineplot(x=x_axis, y=y_axis, data=df_o.loc[df_o['lambda'] == 'basic'])

# Add points for basic DCN models
g = sns.scatterplot(x=x_axis, y=y_axis, size=z_axis, hue=z_axis, edgecolor='black',
                sizes=(10, 100), hue_norm=(min_z, max_z), size_norm=(min_z, max_z),
                legend=False, marker='o',
                s=50, alpha=0.7, data=df_o.loc[df_o['lambda'] == 'basic'])

# Add points for transferred DCN models
g = sns.scatterplot(x=x_axis, y=y_axis, hue=z_axis, size=z_axis, edgecolor='gray',
                sizes=(10, 100), size_norm=(min_z, max_z), hue_norm=(min_z, max_z),
                legend='brief', marker='o',
                s=50, alpha=0.5, data=df_o.loc[df_o['lambda'] != 'basic'])

# Add annotations next to transferred DCN models
for pid in range(0, len(df_o)):
    if df_o.loc[pid, x_axis] < x_limits[-1]:

        # Show annotations as: lambda [(ssim diff)]
        # The ssim diff is shown if abs() > 0.005 and color depends on the
        # sign of the difference.
        label_color = 'black'
        dcn = df_o.dcn_model[pid].split('/')[0]
        lbda = df_o.dcn_model[pid].split('/')[-1]
        if lbda != 'basic':
            # Find base SSIM
            base_ssim = df_o.loc[df_o['dcn_model'] == '{}/basic'.format(dcn), z_axis].values[0]
            ssim_diff = df_o.loc[pid, z_axis] - base_ssim
            if np.abs(ssim_diff) > 0.005:
                label = '{}\n ({:+.3f})'.format(lbda.strip('0'), ssim_diff)
            else:
                label = lbda.strip('0')

            if ssim_diff < 0:
                label_color = '#990000'
            if ssim_diff > 0:
                label_color = '#004400'

            if y_axis == 'ssim' and lbda != '0.0010':
                continue
        else:
            label = '{}\n ({:.2f})'.format(lbda, base_ssim)

        if y_axis == 'accuracy':
            label = label.replace('\n', '')
            ha = 'left'
        else:
            ha = 'center'

        # Draw the actual label
        ax.text(df_o.loc[pid, x_axis] + x_shift, df_o.loc[pid, y_axis] - y_shift, label,
                horizontalalignment=ha, size=7, color=label_color,
                ha=ha)

# Add lines & points for JPEG -------------------------------------------------
g = sns.lineplot(x=x_axis, y=y_axis, data=df_j, dashes=True)

g = sns.scatterplot(x=x_axis, y=y_axis, hue=z_axis, size=z_axis, edgecolor='gray',
                sizes=(10, 100), size_norm=(min_z, max_z), hue_norm=(min_z, max_z),
                legend=False, marker='D', s=50, alpha=0.7, data=df_j)

for pid in range(0, len(df_j)):
    if df_j.loc[pid, x_axis] < x_limits[-1] and df_j.loc[pid, y_axis] > y_limits[0] and int(df_j.quality[pid]) % 10 == 0:
        ax.text(df_j.loc[pid, x_axis] + 0.035, df_j.loc[pid, y_axis], 'JPG({})'.format(int(df_j.quality[pid])),
             horizontalalignment='left', size=7, color='black', va='center')

# Final touches ---------------------------------------------------------------
for line in g.lines:
    line.set_linestyle(':')

# Sometimes seaborn messes up legend entries - number of decimal places explodes
for t in ax.legend_.texts:
    try:
        t.set_text('{:.2f}'.format(float(t.get_text())))
    except:
        pass

g.figure.set_size_inches((7, 5))
ax.grid(True, linestyle=':')
ax.set_xlim(x_limits)
ax.set_ylim(y_limits)
# ax.set_title('{} dataset'.format(dataset.strip('/').split('/')[-1]))
ax.set_xlabel(axis_labels[x_axis])
ax.set_ylabel(axis_labels[y_axis])
# ax.legend(loc='lower right', fancybox=True, framealpha=0.5)
if y_axis == 'ssim':
    ax.set_title('(b) rate-distortion trade-off')
if y_axis == 'accuracy':
    ax.set_title('(a) rate-accuracy trade-off')


fig.tight_layout()
fig.savefig('fig_summary_{}_vs_{}_{}.pdf'.format(y_axis, x_axis, dataset.strip('/').split('/')[-1]),
            dpi=100)

# %%

df = pd.read_csv(os.path.join('../results/', 'summary-dcn-all-7m-renamed.csv'), index_col=False)
df = df.sort_values('scenario')

lookup = ['codec', 'configuration']

# Guess scenario
components = df['scenario'].str.split("/", expand=True)
for i in components:
    df[lookup[i]] = components[i]

df['scenario'] = coreutils.remove_commons(df['scenario'])

mapping = {}
mapping_targets = ['col', 'col', 'hue', 'style', 'size']
mapping_id = 0

# Choose the feature with most unique values as x axis
uniques = [len(df[lookup[i]].unique()) for i in components]

x_feature = np.argmax(uniques)

for i in components:
    if i == x_feature:
        continue

    if len(df[lookup[i]].unique()) > 1:
        mapping[mapping_targets[mapping_id]] = lookup[i]
        mapping_id += 1


# %%
g = sns.catplot(x=lookup[x_feature], y='accuracy', data=df, kind='box', sharex=False, **mapping)
g.axes[0,0].set_xlim([-0.85, 7.85])
g.axes[0,1].set_xlim([-0.85, 7.85])
g.axes[0,2].set_xlim([-0.85, 7.85])
g.axes[0,3].set_xlim([7.15, 25.85])

g.fig.set_size_inches((18, 4))

g.fig.savefig('fig_dcn_jpeg_accuracy.pdf', dpi=100)

# sns.catplot(x='scenario:0', y='dcn_ssim', data=df, kind='box', **mapping)
# sns.scatterplot(x='dcn_ssim', y='accuracy', data=df)
# plt.show()
