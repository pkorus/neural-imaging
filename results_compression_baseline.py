#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 12:19:20 2019

@author: pkorus
"""
from helpers import plotting
from compression import ratedistortion

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

    ratedistortion.plot_curve(plots, axes[j * (n_images+1)], title='{}-bit repr.'.format(bpf), images=[])
    for i, im in enumerate(images):
        ratedistortion.plot_curve(plots, axes[j * (n_images+1) + i + 1], title='Example {}'.format(im), images=[im])

# %% Entropy-regularization
# I. Fix codebook and see impact of regularization and #features

plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-5bpf'}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
# plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-3bpf', 'entropy_reg': 250}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((18, 10))
ratedistortion.plot_curve(plots, axes[0], title='3-bpf codebook w. var reg/#features', images=[])
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], title='Example {}'.format(im), images=[im])

# %%

# plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-3bpf'}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]
plots = [('dcn-entropy.csv', {'quantization': 'soft-codebook-5bpf', 'entropy_reg': 250}), ('jpeg.csv', {}), ('jpeg2000.csv', {})]

images = [0, 11, 13, 30, 36]

fig, axes = plotting.sub(len(images)+1, ncols=3)
fig.set_size_inches((15, 8))
ratedistortion.plot_curve(plots, axes[0], title='DCN with binary repr.', images=[])
for i, im in enumerate(images):
    ratedistortion.plot_curve(plots, axes[i+1], title='Example {}'.format(im), images=[im])
