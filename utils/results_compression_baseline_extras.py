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
from collections import OrderedDict

import seaborn as sns
from matplotlib import rc
import matplotlib.pyplot as plt

sns.set('paper', font_scale=1, style="darkgrid")
sns.set_context("paper")
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


from test_dcn import match_jpeg
from helpers import plotting, loading, utils
from compression import ratedistortion, afi
from training import compression

dataset = '../data/clic512'

def get_sample_images(dataset):

    if 'clic' in dataset:
        return [0, 13, 33, 36]
    
    if 'kodak' in dataset:
        return [4, 14, 20, 22]
    
    if 'raw' in dataset:
        return [11, 19, 34, 35]

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
