#%%
import os
import sys
sys.path.append('..')

import numpy as np
import seaborn as sns
from matplotlib import rc
from sklearn import manifold

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=False)
# sns.set()
# sns.set_context("paper")
# sns.set('paper', font_scale=1, style="ticks")

import matplotlib.pyplot as plt

from helpers import plotting, dataset, utils
from training.manipulation import construct_models, train_manipulation_nip

import cooccurrences

# %%

use_compressed = False
nip_model = 'DNet'
trainable = set()

# Define the distribution channel
distribution = {
    'downsampling': 'none',
    'compression': 'dcn',
    'compression_params': {
        # 'dirname': '../data/raw/dcn/forensics/8k-0.0010', # 95% accuracy
        # 'dirname': '../data/raw/dcn/forensics/8k-0.0050', # 89% accuracy
        # 'dirname': '../data/raw/dcn/forensics/8k-0.0100', # 85% accuracy
        # 'dirname': '../data/raw/dcn/forensics/8k-0.0500', # 72% accuracy
        'dirname': '../data/raw/dcn/forensics/8k-basic',  # 62% accuracy
    }
}

if use_compressed:
    model_name = os.path.split(distribution['compression_params']['dirname'])[-1]
else:
    model_name = 'uncompressed'

# Construct the TF model
tf_ops, distribution = construct_models(nip_model, patch_size=64, trainable=trainable, distribution=distribution, loss_metric='L2')

# %%

# Test the constructed pipeline - show various post-processed versions of a patch

root = '../data/raw'
camera_name = 'D90'
patch_size = 64

data = dataset.IPDataset(os.path.join(root, 'nip_training_data/', camera_name),
                         n_images=0,
                         v_images=64,
                         load='xy',
                         val_rgb_patch_size=2 * patch_size,
                         val_n_patches=1)

# Load the camera model
tf_ops['nip'].load_model(os.path.join(root, 'nip_model_snapshots', camera_name))
if 'dirname' in distribution['compression_params']: tf_ops['dcn'].load_model(distribution['compression_params']['dirname'])

# %% Show sample image + manipulated versions

n_classes = len(distribution['forensics_classes'])
n_batch = 64
show_batch = 16

sample_x, sample_y = data.next_validation_batch(0, n_batch)

# Run the patch through the network
if model_name == 'uncompressed':
    y_patches = tf_ops['sess'].run(tf_ops['operations'], feed_dict={tf_ops['nip'].x: sample_x})
else:
    y_patches = tf_ops['sess'].run(tf_ops['fan'].x, feed_dict={tf_ops['nip'].x: sample_x})

if isinstance(y_patches, np.ndarray):
    sample_c = np.zeros((n_classes * sample_y.shape[0], *sample_y.shape[1:]), np.float32)
    for i in range(sample_y.shape[0]):
        for v in range(n_classes):
            sample_c[i*n_classes + v] = y_patches[v * sample_y.shape[0] + i]

elif isinstance(y_patches, tuple):
    sample_c = np.zeros((n_classes * sample_y.shape[0], *sample_y.shape[1:]), np.float32)
    for i in range(sample_y.shape[0]):
        for v in range(n_classes):
            sample_c[i*n_classes + v] = y_patches[v][i]


# Select images with some interesting content
indices = np.argsort(np.var(sample_x[:,:,:,0], axis=(1,2)))[::-1][:show_batch]

c_indices = np.zeros((show_batch * n_classes), np.uint32)
for i in range(show_batch):
    for c in range(n_classes):
        c_indices[i * n_classes + c] = indices[i] * n_classes + c

# Plot the images
fig = plotting.imsc(sample_c[c_indices],
                    show_batch * ['{}: ()'.format(x) for x in distribution['forensics_classes']],
                    ncols=2 * len(distribution['forensics_classes']),
                    figwidth=20)

fig.savefig('images_{}.pdf'.format(model_name), bbox_inches='tight', dpi=150)

# %% Show residuals

import scipy as sp

def nm(x):
    return (x - x.min()) / (x.max() - x.min())

gk = cooccurrences.filters['f']

sample_r = np.copy(sample_c)

for i in range(sample_r.shape[0]):
    for c in range(sample_r.shape[-1]):
        sample_r[i, :, :, c] = nm(np.abs(sp.signal.convolve2d(sample_r[i, :, :, c], gk, 'same')))

fig = plotting.imsc(sample_r[c_indices],
                    show_batch * ['{}: ()'.format(x) for x in distribution['forensics_classes']],
                    ncols=2 * len(distribution['forensics_classes']),
                    figwidth=20)

fig.savefig('residuals_{}.pdf'.format(model_name), bbox_inches='tight', dpi=150)

# %% Compute sub-band energies

from scipy.fftpack import dct

def mask(size=128, band=0.1, sigma=1):
    x = np.arange(size).reshape((-1, 1)).repeat(size, axis=1)
    y = np.arange(size).reshape((1, -1)).repeat(size, axis=0)
    m = np.exp(-sigma * np.abs(np.power(x + y - band * (size*2 - 1), 2)))
    m[0, 0] = 0
    m = m / m.sum()
    return m

bands = np.linspace(0, 1, 256)
sample_dct = np.copy(sample_c)

energy = np.zeros((sample_dct.shape[0] // n_classes, n_classes, len(bands)))

for i in range(sample_dct.shape[0]):
    for c in range(sample_dct.shape[-1]):
        dct = np.abs(sp.fftpack.dct(sp.fftpack.dct(sample_dct[i, :, :, c].T).T))

        for b in range(len(bands)):
            m = mask(sample_dct.shape[1], bands[b], 1e-2)
            energy[i // n_classes, i % n_classes, b] += np.mean(dct * m) / sample_dct.shape[-1]

        sample_dct[i, :, :, c] = nm(np.log(1 + dct))

fig = plotting.imsc(sample_dct[c_indices],
                    show_batch * ['{}: ()'.format(x) for x in distribution['forensics_classes']],
                    ncols=2 * len(distribution['forensics_classes']),
                    figwidth=20)

# %% Show energy in frequency sub-bands

# fig, axes = plt.subplots(ncols=n_classes, sharey=True, figsize=(n_classes*5, 3))
# for c in range(n_classes):
#     axes[c].semilogy(bands, np.mean(energy, axis=0)[c, :], 'o-')

fig = plt.figure()
axes = plt.gca()
for c in range(n_classes):
    axes.semilogy(bands, np.mean(energy, axis=0)[c, :], '-')

plt.legend(distribution['forensics_classes'])
axes.set_xlabel('Frequency sub-band [relative range]')
axes.set_ylabel('Energy level')

# %% Residual features

def nm(x):
    return (x - x.min()) / (x.max() - x.min())

gk = np.array([[-0.0833, -0.1667, -0.0833], [-0.1667, 1.0000, -0.1667], [-0.0833, -0.1667, -0.0833]])

# codebook = np.linspace(0, 1, 256)
codebook = np.linspace(-0.5, 0.5, 256)

filters = cooccurrences.filters

y = np.zeros((3 * sample_c.shape[0]))
X = np.zeros((3 * sample_c.shape[0], 2 * 81))

residuals = {k: np.zeros((sample_c.shape[0] // n_classes, n_classes, len(codebook))) for k in filters.keys()}

for i in range(sample_c.shape[0]):
    for c in range(sample_c.shape[-1]):

        # Forensics co-occurrences features (separate for RGB channels)
        X[i * 3 + c, :81] = cooccurrences.get_block_features(255 * sample_c[i, :, :, c], f='f')
        X[i * 3 + c, 81:] = cooccurrences.get_block_features(255 * sample_c[i, :, :, c].T, f='f')
        y[i * 3 + c] = i % n_classes

        # Histograms of residual features (averaged over RGB channels)
        for f_label, f in filters.items():
            val = sp.signal.convolve2d(sample_c[i, :, :, c], f, 'same')
            if codebook.min() >= 0:
                val = np.abs(val)
            residuals[f_label][i // n_classes, i % n_classes] += utils.qhist(val, codebook) / sample_dct.shape[-1]

fig = plt.figure()
axes = plt.gca()
key = 'h'
for c in range(n_classes):
    axes.semilogy(codebook, residuals[key].mean(axis=0)[c, :], '-')
axes.legend(distribution['forensics_classes'])
axes.set_xlabel('{}-residual value'.format(key))
axes.set_ylabel('Frequency')
axes.set_yticks([])
if codebook.min() >= 0:
    axes.set_xlim([-0.0075, 0.25])
else:
    axes.set_xlim([-0.25, 0.25])
axes.set_ylim([1e-1, 2 * residuals[key].mean(axis=0).max()])

# %%

plot_keys = ['h', 'v']

fig, axes = plt.subplots(ncols=len(plot_keys) + 2, figsize=(4*(2 + len(plot_keys)), 3))

for c in range(n_classes):
    axes[0].semilogy(bands, np.mean(energy, axis=0)[c, :], '-')

axes[0].legend(distribution['forensics_classes'])
axes[0].set_xlabel('Frequency sub-band [relative range]')
axes[0].set_ylabel('Energy level')

# t-SNE
tsne = manifold.TSNE(n_components=2, init='random', random_state=0, perplexity=50)
Y = tsne.fit_transform(X)

for c in range(5):
    ind = y == c
    axes[1].scatter(Y[ind, 0], Y[ind, 1], alpha=0.25)
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].set_title('t-SNE of co-occurrence features')

# Distributions of residuals
for i, key in enumerate(plot_keys):
    for c in range(n_classes):
        axes[2 + i].semilogy(codebook, residuals[key].mean(axis=0)[c, :], '-')
    # axes[1 + i].legend(distribution['forensics_classes'])
    axes[2 + i].set_xlabel('{}-residual value'.format(key))
    axes[2 + i].set_ylabel('Frequency')
    axes[2 + i].set_yticks([])
    if codebook.min() >= 0:
        axes[2 + i].set_xlim([-0.0075, 0.25])
    else:
        axes[2 + i].set_xlim([-0.25, 0.25])
    axes[2 + i].set_ylim([1e-1, 2 * residuals[key].mean(axis=0).max()])

fig.savefig('summary_{}.pdf'.format(model_name), bbox_inches='tight')

# %% t-SNE of co-occurrence features

import coocurrences

residuals = np.zeros((n_classes, sample_c.shape[0] // n_classes, 3, 2 * 81))
labels = np.zeros((3 * sample_c.shape[0]))
X = np.zeros((3 * sample_c.shape[0], 2 * 81))

for i in range(sample_c.shape[0]):
    for c in range(sample_c.shape[-1]):
        residuals[i % n_classes, i // n_classes, c, :81] = cooccurrences.get_block_features(255 * sample_c[i, :, :, c], f='f')
        residuals[i % n_classes, i // n_classes, c:, 81:] = cooccurrences.get_block_features(255 * sample_c[i, :, :, c].T, f='f')
        X[i * 3 + c] = residuals[i % n_classes, i // n_classes, c]
        labels[i * 3 + c] = i % n_classes

# Show the t-SNE embedding

from sklearn import manifold

tsne = manifold.TSNE(n_components=2, init='random', random_state=2, perplexity=50)
Y = tsne.fit_transform(X)

for c in range(5):
    ind = labels == c
    plt.scatter(Y[ind, 0], Y[ind, 1], alpha=0.25)

# %%

plt.imshow(X[labels == 0])