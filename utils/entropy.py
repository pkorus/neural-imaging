#%%
import sys
sys.path.append('..')

import numpy as np

# import matplotlib
from matplotlib import rc
# rc('font',family='serif')
rc('text', usetex=True)
# rc('xtick', labelsize='x-small')
# rc('ytick', labelsize='x-small')

import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
sns.set_context("paper")

#%%

def quantize(x, codebook, v=100, sigma=5, dtype=np.float64):
    eps = 1e-72
    
    codebook = codebook.reshape((1, -1)).astype(dtype)
    values = x.reshape((-1, 1)).astype(dtype)

    if v <= 0:
        # Gaussian soft quantization
        weights = np.exp(-sigma * np.power(values - codebook, 2))
    else:
        # t-Student soft quantization
        dff = sigma * (values - codebook)
        weights = np.power((1 + np.power(dff, 2)/v), -(v+1)/2)
    
    weights = (weights + eps) / (eps + np.sum(weights, axis=1, keepdims=True))
    
    assert(weights.shape[1] == np.prod(codebook.shape))

    soft = np.matmul(weights, codebook.T)
    soft = soft.reshape(x.shape)            

    hard = codebook.reshape((-1,))[np.argmax(weights, axis=1)]
    hard = hard.reshape(x.shape)     
    
    histogram = np.mean(weights, axis=0)
    histogram = np.clip(histogram, 1e-9, np.finfo(np.float64).max)
    histogram = histogram / np.sum(histogram)
    entropy = - np.sum(histogram * np.log2(histogram))

    return soft, hard, histogram, entropy, weights


def quantize_real(x, codebook):
    X_rnd = np.round(X).clip(-c_max, c_max)
    hist = np.zeros_like(codebook)
    unique, counts = np.unique(X_rnd.astype(np.int), return_counts=True)
    indices = np.where(np.abs(codebook.reshape((-1,1)) - unique.reshape((1,-1))) == 0)[0]
    hist[indices] = counts
    hist = hist.clip(1)
    hist = hist / hist.sum()
    entropy_real = - np.sum(hist * np.log2(hist))

    return X_rnd, hist, entropy_real


#%% Single example

c_max = 5

# Generate random data
# X = 3 * np.random.normal(size=(10000,1))
X = np.random.laplace(size=(2000,1), scale=2)

codebook = np.arange(-c_max, c_max+1, 1)

# Standard rounding / histogram / entropy
X_rnd, hist, entropy_real = quantize_real(X, codebook)

# Soft approximations
X_soft, X_hard, histogram, entropy, weights = quantize(X, codebook, v=50, sigma=5)

fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(20,12))
axes[0, 0].plot(X_rnd, X_hard, '.')
axes[0, 0].plot([-c_max, c_max], [-c_max, c_max], ':')
axes[0, 0].set_title('Standard vs estimated hard quantization')
axes[0, 0].set_xlabel('standard rounding')
axes[0, 0].set_ylabel('hard estimate')


axes[0, 1].plot(X, X_soft, '.')
axes[0, 1].plot(X, X_rnd, '.')
axes[0, 1].set_title('Soft quantization vs input')
axes[0, 1].legend(['soft estimate', 'real quantization'])

axes[1, 0].plot(codebook, histogram, '.-')
axes[1, 0].plot(codebook, hist, '.-')
axes[1, 0].set_title('Histograms: real vs estimated')
axes[1, 0].legend(['soft estimate', 'real histogram'])

axes[1, 1].plot(histogram, hist, '.')
axes[1, 1].plot([0, 1], [0, 1], ':')
axes[1, 1].set_xlim([0.9*hist.min(), 1.05*hist.max()])
axes[1, 1].set_ylim([0.9*hist.min(), 1.05*hist.max()])
axes[1, 1].set_title('Histogram bins: real vs estimated')

axes[0, 2].imshow(weights[0:c_max*3], cmap='gray')
axes[0, 2].grid(False)
axes[1, 2].remove()

quant_h_error = np.mean(np.abs(X_rnd - X_hard))
quant_s_error = np.mean(np.abs(X_rnd - X_soft))
hist_error = np.mean(np.abs(histogram - hist))
kld = - np.sum(hist * np.log2(histogram / hist))

print('Quantization error (hard) : {:.4f}'.format(quant_h_error))
print('Quantization error (soft) : {:.4f}'.format(quant_s_error))
print('Histogram bin error       : {:.4f}'.format(hist_error))
print('Entropy                   : {:.4f}'.format(entropy_real))
print('Entropy (soft)            : {:.4f}'.format(entropy))
print('Entropy error             : {:.2f}'.format(np.abs(entropy_real - entropy)))
print('Entropy error             : {:.3f}%'.format(100 * np.abs(entropy_real - entropy) / entropy_real))
print('Kullback-Leibler div.     : {:.4f}'.format(kld))

#%% Histogram & quantization for various distribution scales

v = 0
sigma = 5
c_max = 5
n_samples = 1000
distribution = 'Gaussian'

codebook = np.arange(-c_max, c_max+1, 1)

fig, axes = plt.subplots(2, 5, squeeze=False, figsize=(20,6))

for i, scale in enumerate([0.15, 0.5, 1, 2, 4]):

    if distribution == 'Laplace':
        X = np.random.laplace(size=(n_samples, 1), scale=scale)
    elif distribution == 'Gaussian':
        X = scale * np.random.normal(size=(n_samples, 1))

    # Standard rounding / histogram / entropy
    X_rnd, hist, entropy_real = quantize_real(X, codebook)
    
    # Soft approximations
    X_soft, X_hard, histogram, entropy, weights = quantize(X, codebook, v=v, sigma=sigma)

    axes[0, i].plot(X, X_soft, '.')
    axes[0, i].plot(X, X_rnd, '.')
    axes[0, i].set_title('Soft quantization vs input')
    axes[0, i].legend(['soft estimate', 'real quantization'])
    axes[0, i].set_title('{} dist. s={}; Kernel: {}, $\sigma$={}'.format(distribution, scale, 'Gaussian' if v == 0 else 't-Student({})'.format(v), sigma))
    if i == 0:
        axes[0, i].set_ylabel('Quantized values')

    axes[1, i].plot(codebook, histogram, '.-')
    axes[1, i].plot(codebook, hist, '.-')
    # axes[1, i].set_title('Histograms: real vs estimated')
    axes[1, i].legend(['soft (H={:.2f})'.format(entropy), 'real (H={:.2f})'.format(entropy_real)])
    if i == 0:
        axes[1, i].set_ylabel('Histograms')

fig.savefig('fig_quantization_n_hist_{}.pdf'.format(distribution), bbox_inches='tight')

#%%

def estimate_errors(X, codebook, v=100, sigma=5):
    # Standard rounding / histogram / entropy
    _, _, hard_entropy = quantize_real(X, codebook)
    
    # Soft approximations
    _, _, _, soft_entropy, _ = quantize(X, codebook, v, sigma)

    entropy_error = np.abs(hard_entropy - soft_entropy)
    
    return hard_entropy, soft_entropy, entropy_error

#%% Large synthetic data experiment

v = 0
sigma = 5
n_scales = 500
n_samples = 1000
distribution = 'Laplace'
codebook = np.arange(-c_max, c_max+1, 1)

data = np.zeros((5, n_scales))
data[0] = np.linspace(0.01, 10, n_scales)

for i, scale in enumerate(data[0]):
    
    if distribution == 'Laplace':
        X = np.random.laplace(size=(n_samples, 1), scale=scale)
    elif distribution == 'Gaussian':
        X = scale * np.random.normal(size=(n_samples, 1))
        
    data[1:-1, i] = estimate_errors(X, codebook, v, sigma)

data[-1] = 100 * data[3] / data[1]

fig, axes = plt.subplots(1, 3, squeeze=False, figsize=(15,4))
axes[0, 0].plot(data[0], data[3], '.', alpha=0.5, markersize=3)
axes[0, 0].set_xlabel('{} distribution scale'.format(distribution))
axes[0, 0].set_ylabel('Absolute entropy error')

axes[0, 1].plot(data[0], data[4], '.', alpha=0.5, markersize=3)
axes[0, 1].set_xlabel('{} distribution scale'.format(distribution))
axes[0, 1].set_ylabel('Relative entropy error [\\%]')

axes[0, 2].plot(data[1], data[2], '.', alpha=0.5, markersize=5)
axes[0, 2].plot([0, 5], [0, 5], ':')
axes[0, 2].set_xlim([-0.05, 1.05*max(data[1])])
axes[0, 2].set_ylim([-0.05, 1.05*max(data[1])])
axes[0, 2].set_xlabel('Real entropy')
axes[0, 2].set_ylabel('Soft estimate')

fig.suptitle('{} dist.; Kernel: {}, $\sigma$={}'.format(distribution, 'Gaussian' if v == 0 else 't-Student({})'.format(v), sigma))

fig.savefig('fig_errors_{}.pdf'.format('Gaussian' if v == 0 else 't-Student({})'.format(v)), bbox_inches='tight')

#%% Large synthetic data experiment (Alternative)

v = 0
sigma = 5
n_scales = 500
n_samples = 1000
distribution = 'Laplace'
codebook = np.arange(-c_max, c_max+1, 1)

data = np.zeros((5, n_scales))
data[0] = np.linspace(0.01, 10, n_scales)

fig, axes = plt.subplots(1, 4, squeeze=False, figsize=(18, 3))

for v, color in zip([0, 25], ['r', 'g']):

    data = np.zeros((5, n_scales))
    data[0] = np.linspace(0.01, 10, n_scales)

    for i, scale in enumerate(data[0]):

        if distribution == 'Laplace':
            X = np.random.laplace(size=(n_samples, 1), scale=scale)
        elif distribution == 'Gaussian':
            X = scale * np.random.normal(size=(n_samples, 1))
            
        data[1:-1, i] = estimate_errors(X, codebook, v, sigma)
        data[-1] = 100 * data[3] / data[1]

    axes[0, 0].plot(data[0], data[3], 'o', alpha=0.25, markersize=3, color=color)
    axes[0, 0].set_xlabel('{} distribution scale'.format(distribution))
    axes[0, 0].set_ylabel('Absolute entropy error')

    axes[0, 1].plot(data[0], data[4], 'o', alpha=0.25, markersize=3,color=color)
    axes[0, 1].set_xlabel('{} distribution scale'.format(distribution))
    axes[0, 1].set_ylabel('Relative entropy error [\\%]')

    axes[0, 2 + np.sign(v)].plot(data[1], data[2], '.', alpha=0.5, markersize=5, color=color)
    axes[0, 2 + np.sign(v)].plot([0, 5], [0, 5], ':')
    axes[0, 2 + np.sign(v)].set_xlim([-0.05, 1.05*max(data[1])])
    axes[0, 2 + np.sign(v)].set_ylim([-0.05, 1.05*max(data[1])])
    axes[0, 2 + np.sign(v)].set_xlabel('Real entropy')
    axes[0, 2 + np.sign(v)].set_ylabel('Soft estimate')

axes[0, 0].legend(['Gaussian', 't-Student({})'.format(25)])

fig.suptitle('{} dist.; Kernel: {}, $\sigma$={}'.format(distribution, 'Gaussian' if v == 0 else 't-Student({})'.format(v), sigma))

fig.savefig('fig_errors.pdf', bbox_inches='tight')

# %% Hyper-parameter search

n_scales = 250
n_samples = 1000
distribution = 'Gaussian'

vs = [0, 5, 10, 25, 50, 100]
sig = [5, 10, 25, 50]

fig, axes = plt.subplots(len(sig), len(vs), sharex=True, sharey=True,
                         squeeze=False, figsize=(4 * len(vs), 3 * len(sig)))

for n, v in enumerate(vs):
    for m, s in enumerate(sig):

        data = np.zeros((5, n_scales))
        data[0] = np.linspace(0.01, 10, n_scales)
        
        for i, scale in enumerate(data[0]):
            
            if distribution == 'Laplace':
                X = np.random.laplace(size=(n_samples, 1), scale=scale)
            elif distribution == 'Gaussian':
                X = scale * np.random.normal(size=(n_samples, 1))

            data[1:-1, i] = estimate_errors(X, codebook, v, s)

        data[-1] = 100 * data[3] / data[1]

        axes[m, n].plot(data[0], data[4], '.', alpha=0.25, markersize=5)
        if s == sig[-1]:
            axes[m, n].set_xlabel('{} distribution scale'.format(distribution))
        if n == 0:
            axes[m, n].set_ylabel('Relative entropy error [\\%]')
        axes[m, n].set_title('Kernel: {}, $\sigma$={} $\\rightarrow$ {:.2f}'.format('Gaussian' if v == 0 else 't-Student({})'.format(v), s, np.mean(data[4])))

fig.savefig('fig_entropy_hp_{}.pdf'.format(distribution), bbox_inches='tight')

# %% Real compression model and real images

from compression import afi
from helpers import dataset, utils

dcn_presets = {
    '4k': '../data/raw/dcn/entropy/TwitterDCN-4096D/16x16x16-r:soft-codebook-Q-5.0bpf-S+-H+250.00',
    '8k': '../data/raw/dcn/entropy/TwitterDCN-8192D/16x16x32-r:soft-codebook-Q-5.0bpf-S+-H+250.00',
    '16k': '../data/raw/dcn/entropy/TwitterDCN-16384D/16x16x64-r:soft-codebook-Q-5.0bpf-S+-H+250.00'
}

# %%

data = dataset.IPDataset('../data/clic256', n_images=35, v_images=0, load='y')
dcn = afi.restore_model(dcn_presets['8k'])

# %%

n_epochs = data.count_training * 5

results = np.zeros((2, n_epochs))

for epoch in range(n_epochs):

    batch_x = data.next_training_batch(epoch % data.count_training, 1, 128)
    batch_z = dcn.compress(batch_x)
    results[0, epoch] = utils.entropy(batch_z, dcn.get_codebook())
    results[1, epoch] = dcn.sess.run(dcn.entropy, feed_dict={dcn.x: batch_x})

# %%

plt.plot(results[0], results[1], '.', alpha=0.1, markersize=5)
plt.plot([0, 5], [0, 5], ':')
plt.xlim([-0.05, 1.05*max(results[1])])
plt.ylim([-0.05, 1.05*max(results[1])])
plt.xlabel('Real entropy')
plt.ylabel('Soft estimate')
plt.title('Real images + {}'.format(dcn.model_code))
