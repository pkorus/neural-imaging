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

dataset = '../data/kodak512'
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

dcn_model = '4k'

# Define the distribution channel
models = OrderedDict()
models[0.001]   = '../data/raw/dcn/forensics/{}-0.0010'.format(dcn_model) # 95% accuracy
models[0.005]   = '../data/raw/dcn/forensics/{}-0.0050'.format(dcn_model) # 89% accuracy
models[0.010]   = '../data/raw/dcn/forensics/{}-0.0100'.format(dcn_model) # 85% accuracy
models[0.050]   = '../data/raw/dcn/forensics/{}-0.0500'.format(dcn_model) # 72% accuracy
models[0.100]   = '../data/raw/dcn/forensics/{}-0.1000'.format(dcn_model) # 65% accuracy
models[1.000]   = '../data/raw/dcn/forensics/{}-1.0000'.format(dcn_model) # 62% accuracy
# models[1.001]   = '../data/raw/dcn/forensics/{}-1.0000b'.format(dcn_model) # 62% accuracy
# models[1.002]   = '../data/raw/dcn/forensics/{}-1.0000c'.format(dcn_model) # 62% accuracy
models[5.000]   = '../data/raw/dcn/forensics/{}-5.0000'.format(dcn_model) # 62% accuracy
# models[1000.0]   = '../data/raw/dcn/forensics/{}-1000.0'.format(dcn_model) # 62% accuracy
models['basic'] = '../data/raw/dcn/forensics/{}-basic'.format(dcn_model)  # 62% accuracy

outputs = OrderedDict()
stats = OrderedDict()

# Worst images for clic: 1, 28, 33, 36
image_id = 34 # 32 # 28 for clic

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

# # Dump to file
for key, value in outputs.items():
    os.makedirs('debug/{}/{}/'.format(dataset.strip('/').split('/')[-1], image_id), exist_ok=True)
    imageio.imwrite('debug/{}/{}/{}_{}.png'.format(dataset.strip('/').split('/')[-1], image_id, dcn_model, key), value.squeeze())
    with open('debug/{}/{}/{}_log.txt'.format(dataset.strip('/').split('/')[-1], image_id, dcn_model), 'w') as f:
        f.write('# {}\n'.format(files[image_id]))
        for model in models.keys():
            f.write('{:>10} : ssim = {:.3f} @ {:.3f} bpp\n'.format(model, stats[model]['ssim'][0], stats[model]['bpp'][0]))

# %%

fig = plotting.imsc([nm(outputs[k] - outputs['basic']) for k in outputs.keys()],
                    ['{}'.format(x) for x in models.keys()], figwidth=24)

fig.savefig('diff.pdf', bbox_inches='tight')

# %% Images

from diff_nip import compare_images_ab_ref

use_pretrained_ref = False

if use_pretrained_ref:
    reference = outputs['basic']
else:
    reference = batch_x[image_id:image_id+1]

fig = compare_images_ab_ref(reference, outputs[1.000], outputs[0.001],
                            labels=['Pre-trained DCN' if use_pretrained_ref else 'Original image', '$\lambda=1.000$', '$\lambda=0.001$'])

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

# %% Perception Distance (AlexNet)

from models import lpips

distance = lpips.lpips(outputs['basic'], batch_x[image_id:image_id+1])

print(distance)


# %% Pool compression quality and classification accuracy

dataset = '../data/raw512/'
dcn_models = ['4k', '8k', '16k']
lambdas = [0.001, 0.005, 0.010, 0.050, 0.100, 1.000, 'basic']

models = OrderedDict()

# --- DCNs --------------------------------------------------------------------
df = pd.read_csv(os.path.join(dataset, 'dcn-forensics.csv'), index_col=False)
df_acc = pd.read_csv(os.path.join('../results/', 'summary-dcn-all.csv'), index_col=False)

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
df_acc = pd.read_csv(os.path.join('../results/', 'summary-jpeg.csv'), index_col=False)

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


# %%

fig = plt.figure()
ax = fig.gca()

x_max = 1.0

min_ssim = 0.875
max_ssim = max([df_j['ssim'].max(), df_o['ssim'].max()])

g = sns.lineplot(x='bpp', y='accuracy', data=df_o.loc[df_o['lambda'] == 'basic'])

g = sns.scatterplot(x='bpp', y='accuracy', size='ssim', hue='ssim', edgecolor='black',
                sizes=(20, 100), hue_norm=(min_ssim, max_ssim), size_norm=(min_ssim, max_ssim),
                legend=False, marker='o',
                s=50, alpha=0.7, data=df_o.loc[df_o['lambda'] == 'basic'])

g = sns.scatterplot(x='bpp', y='accuracy', hue='ssim', size='ssim', edgecolor='white',
                sizes=(20, 100), size_norm=(min_ssim, max_ssim), hue_norm=(min_ssim, max_ssim),
                legend='brief', marker='o',
                s=50, alpha=0.7, data=df_o.loc[df_o['lambda'] != 'basic'])

for pid in range(0, len(df_o)):
    if df_o.bpp[pid] < x_max:
        ax.text(df_o.bpp[pid] + 0.015, df_o.accuracy[pid], df_o.dcn_model[pid].split('/')[-1],
             horizontalalignment='left', size=7, color='black')

g = sns.lineplot(x='bpp', y='accuracy', data=df_j, dashes=True)

for line in g.lines:
    line.set_linestyle(':')

g = sns.scatterplot(x='bpp', y='accuracy', hue='ssim', size='ssim', edgecolor='gray',
                sizes=(20, 100), size_norm=(min_ssim, max_ssim), hue_norm=(min_ssim, max_ssim),
                legend=False, marker='D', s=50, alpha=0.7, data=df_j)

for pid in range(0, len(df_j)):
    if df_j.bpp[pid] < x_max and int(df_j.quality[pid]) % 10 == 0:
        ax.text(df_j.bpp[pid] + 0.02, df_j.accuracy[pid], int(df_j.quality[pid]),
             horizontalalignment='left', size=7, color='black')


g.figure.set_size_inches((10, 6))
ax.grid(True, linestyle=':')
ax.set_xlim([0.2, x_max])
ax.legend(loc='lower right', fancybox=True, framealpha=0.5)
ax.set_title(dataset)

# %%

g = sns.scatterplot(x='bpp', y='accuracy', hue='ssim',
                sizes=(20, 100), size_norm=(0.8, 1.0), hue_norm=(0.75, 0.95),
                legend='brief', marker='D', s=50, alpha=0.7, data=df_j)


# %%
markers = {}
markers['basic'] = 's'
for lbda in lambdas:
    if type(lbda) is float:
        markers['{:.4f}'.format(lbda)] = 'o'

g = sns.scatterplot(x='bpp', y='accuracy', hue='lambda', size='ssim', style='lambda',
                sizes=(20, 100), legend='brief', markers=markers,
                s=50, alpha=0.7, data=df_o)

g.figure.set_size_inches((10, 6))
g.axes.grid(True, linestyle=':')


# %%

g = sns.scatterplot(x='bpp', y='accuracy', hue='ssim', size='ssim',
                sizes=(20, 100), legend='brief', marker='2',
                s=50, alpha=0.7, data=df_j)
