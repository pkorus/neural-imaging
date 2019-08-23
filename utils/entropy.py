#%%
import sys
sys.path.append('..')

import numpy as np

import seaborn as sns
from matplotlib import rc

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

import matplotlib.pyplot as plt

#%%

def quantize(x, codebook, v=100, gamma=5, dtype=np.float64):
    eps = 1e-72
    
    codebook = codebook.reshape((1, -1)).astype(dtype)
    values = x.reshape((-1, 1)).astype(dtype)

    if v <= 0:
        # Gaussian soft quantization
        weights = np.exp(-gamma * np.power(values - codebook, 2))
    else:
        # t-Student soft quantization
        dff = (values - codebook)
        weights = np.power((1 + gamma * np.power(dff, 2)/v), -(v+1)/2)
    
    weights = (weights + eps) / (np.sum(weights + eps, axis=1, keepdims=True))
    
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
X = np.random.laplace(size=(1000,1), scale=5)

codebook = np.arange(-c_max, c_max+1, 1)

# Standard rounding / histogram / entropy
X_rnd, hist, entropy_real = quantize_real(X, codebook)

# Soft approximations
X_soft, X_hard, histogram, entropy, weights = quantize(X, codebook, v=50, gamma=1)

indices = np.argsort(X.T).reshape((-1, ))

fig, axes = plt.subplots(2, 3, squeeze=False, figsize=(20,12))
axes[0, 0].plot(X_rnd, X_hard, '.')
axes[0, 0].plot([-c_max, c_max], [-c_max, c_max], ':')
axes[0, 0].set_title('Standard vs estimated hard quantization')
axes[0, 0].set_xlabel('standard rounding')
axes[0, 0].set_ylabel('hard estimate')


axes[0, 1].plot(X[indices], X_soft[indices], '--')
axes[0, 1].plot(X[indices], X_rnd[indices], '-')
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

c_max = 5
n_samples = 10000
distribution = 'Laplace'

codebook = np.arange(-c_max, c_max+1, 1)

fig, axes = plt.subplots(2, 5, squeeze=False, figsize=(20,6))

for i, scale in enumerate([0.15, 0.5, 1, 2, 4]):

    if distribution == 'Laplace':
        X = np.random.laplace(size=(n_samples, 1), scale=scale)
    elif distribution == 'Gaussian':
        X = scale * np.random.normal(size=(n_samples, 1))

    # Standard rounding / histogram / entropy
    X_rnd, hist, entropy_real = quantize_real(X, codebook)

    entropies = [entropy_real]

    for v, gamma in zip([0, 50], [5, 25]):

        # Soft approximations
        X_soft, X_hard, histogram, entropy, weights = quantize(X, codebook, v=v, gamma=gamma)
    
        indices = np.argsort(X.T).reshape((-1, ))
        entropies.append(entropy)

        if v == 0:
            axes[0, i].plot(X[indices], X_rnd[indices], '--', markersize=1)
        axes[0, i].plot(X[indices], X_soft[indices], '-', markersize=1)
        axes[0, i].set_title('Soft quantization vs input')
        axes[0, i].set_title('Random sample: {} $\\lambda$={};'.format(distribution, scale))
        if i == 0:
            axes[0, i].set_ylabel('Quantized values')
    
        axes[1, i].plot(codebook, hist, '.-')
        axes[1, i].plot(codebook, histogram, '.-')
        # axes[1, i].set_title('Histograms: real vs estimated')
        if i == 0:
            axes[1, i].set_ylabel('Histograms')

    axes[0, i].legend(['real quantization', 'soft est. (Gaussian)', 'soft est. (t-Student)'])
    axes[1, i].legend(['{} (H={:.2f})'.format(k, v) for k, v in zip(['real', 'soft/Gaussian', 'soft/t-Student'], entropies)])

fig.savefig('fig_quantization_n_hist_{}.pdf'.format(distribution), bbox_inches='tight')

#%%

def estimate_errors(X, codebook, v=100, gamma=5):
    # Standard rounding / histogram / entropy
    _, _, hard_entropy = quantize_real(X, codebook)
    
    # Soft approximations
    _, _, _, soft_entropy, _ = quantize(X, codebook, v, gamma)

    entropy_error = np.abs(hard_entropy - soft_entropy)
    
    return hard_entropy, soft_entropy, entropy_error

#%% Large synthetic data experiment (single kernel)

v = 50
gamma = 25
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
        
    data[1:-1, i] = estimate_errors(X, codebook, v, gamma)

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

fig.suptitle('{} dist.; Kernel: {}, $\gamma$={}'.format(distribution, 'Gaussian' if v == 0 else 't-Student({})'.format(v), gamma))

fig.savefig('fig_errors_{}.pdf'.format('Gaussian' if v == 0 else 't-Student({})'.format(v)), bbox_inches='tight')

#%% Large synthetic data experiment (both kernels)

c_max = 5
n_scales = 500
n_samples = 1000
distribution = 'Laplace'
codebook = np.arange(-c_max, c_max+1, 1)

data = np.zeros((5, n_scales))
data[0] = np.linspace(0.01, 10, n_scales)

fig, axes = plt.subplots(1, 3, squeeze=False, figsize=(12, 3))

for v, gamma, color in zip([0, 50], [5, 25], ['r', 'g']):

    data = np.zeros((5, n_scales))
    data[0] = np.linspace(0.01, 10, n_scales)

    for i, scale in enumerate(data[0]):

        if distribution == 'Laplace':
            X = np.random.laplace(size=(n_samples, 1), scale=scale)
        elif distribution == 'Gaussian':
            X = scale * np.random.normal(size=(n_samples, 1))
            
        data[1:-1, i] = estimate_errors(X, codebook, v, gamma)
        data[-1] = 100 * data[3] / data[1]

    axes[0, 0].plot(data[0], data[3], 'o', alpha=0.25, markersize=3, color=color, label='Gaussian $\cdotp$ $\gamma$={}'.format(gamma) if v == 0 else 't-Student({}) $\cdotp$ $\gamma$={}'.format(v, gamma))
    axes[0, 0].set_xlabel('{} distribution scale'.format(distribution))
    axes[0, 0].set_ylabel('Absolute entropy error')

    axes[0, 1].plot(data[0], data[4], 'o', alpha=0.25, markersize=3,color=color)
    axes[0, 1].set_xlabel('{} distribution scale'.format(distribution))
    axes[0, 1].set_ylabel('Relative entropy error [\\%]')

    #  + np.sign(v)
    axes[0, 2].plot(data[1], data[2], '.', alpha=0.5, markersize=5, color=color)
    axes[0, 2].plot([0, 5], [0, 5], ':')
    axes[0, 2].set_xlim([-0.05, 1.05*max(data[1])])
    axes[0, 2].set_ylim([-0.05, 1.05*max(data[1])])
    axes[0, 2].set_xlabel('Real entropy')
    axes[0, 2].set_ylabel('Soft estimate')
    # 'Gaussian' if v == 0 else 't-Student({})'.format(v)

# axes[0, 0].legend(['Gaussian', 't-Student({})'.format(gamma)])

axes[0, 0].legend()

# fig.suptitle('Random sample: {} dist.; Kernel: {}'.format(distribution, 'Gaussian / t-Student'))

plt.tight_layout()
fig.savefig('fig_entropy_errors.pdf')

# %% Hyper-parameter search

n_scales = 500
n_samples = 1000
distribution = 'Laplace'

vs = [0, 5, 10, 25, 50, 100]
sig = [1, 3, 5, 10, 25, 50]

fig, axes = plt.subplots(len(sig), len(vs), sharex=True, sharey=True,
                         squeeze=False, figsize=(4 * len(vs), 3 * len(sig)))

errors = np.zeros((len(sig), len(vs)))

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

        errors[m, n] = np.mean(data[4])

        axes[m, n].plot(data[0], data[4], '.', alpha=0.15, markersize=5)
        if s == sig[-1]:
            axes[m, n].set_xlabel('{} distribution scale'.format(distribution))
        if n == 0:
            axes[m, n].set_ylabel('Relative entropy error [\\%]')
        axes[m, n].set_ylim([-4, 104])
        axes[m, n].set_title('Kernel: {}, $\gamma$={} $\\rightarrow$ {:.2f}'.format('Gaussian' if v == 0 else 't-Student({})'.format(v), s, np.mean(data[4])))

best_index = np.unravel_index(np.argmin(errors), errors.shape)

print('Best error : {:.2f}'.format(errors[best_index]))
print('Index      : {}'.format(best_index))
print('Best v     : {}'.format(vs[best_index[1]]))
print('Best gamma : {}'.format(sig[best_index[0]]))

axes[best_index].set_title(axes[best_index].title.get_text(), color='red')
# fig.show()

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

n_epochs = data.count_training * 10

results = {
    't-Student': np.zeros((2, n_epochs)),
    'Gaussian': np.zeros((2, n_epochs))
}

# %%

# TODO Currently this needs to be done by changing hard-coded values in the background
kernel = 'Gaussian'

for epoch in range(n_epochs):

    batch_x = data.next_training_batch(epoch % data.count_training, 1, 128)
    batch_z = dcn.compress(batch_x)
    results[kernel][0, epoch] = utils.entropy(batch_z, dcn.get_codebook())
    results[kernel][1, epoch] = dcn.sess.run(dcn.entropy, feed_dict={dcn.x: batch_x})

# %%

fig = plt.figure(figsize=(4, 3))
ax = fig.gca()

for kernel, color in zip(['Gaussian', 't-Student'], ['r', 'g']):
    ax.plot(results[kernel][0], results[kernel][1], '.', alpha=0.25, markersize=5, label=kernel, color=color)
ax.plot([0, 5], [0, 5], ':')
ax.set_xlim([0, 1.05*max(results[kernel][1])])
ax.set_ylim([0, 1.05*max(results[kernel][1])])
ax.set_xlabel('Real entropy of latent space')
ax.set_ylabel('Soft estimate')
# ax.set_title(dcn.model_code)
# ax.set_title('DCN with {} channels'.format(dcn._h.n_features))

# fig.savefig('fig_entropy_real-images_{}.pdf'.format('t-Student'), bbox_inches='tight')
plt.tight_layout()
fig.savefig('fig_entropy_real-images_{}.pdf'.format('both'))

# %%

#%% Large synthetic data experiment (both kernels)

c_max = 5
n_scales = 500
n_samples = 1000
distribution = 'Laplace'
codebook = np.arange(-c_max, c_max+1, 1)

data = np.zeros((5, n_scales))
data[0] = np.linspace(0.01, 10, n_scales)

fig, axes = plt.subplots(1, 4, squeeze=False, figsize=(16, 3))

for v, gamma, color in zip([0, 50], [5, 25], ['r', 'g']):

    data = np.zeros((5, n_scales))
    data[0] = np.linspace(0.01, 10, n_scales)

    for i, scale in enumerate(data[0]):

        if distribution == 'Laplace':
            X = np.random.laplace(size=(n_samples, 1), scale=scale)
        elif distribution == 'Gaussian':
            X = scale * np.random.normal(size=(n_samples, 1))
            
        data[1:-1, i] = estimate_errors(X, codebook, v, gamma)
        data[-1] = 100 * data[3] / data[1]

    axes[0, 0].plot(data[0], data[3], 'o', alpha=0.25, markersize=3, color=color, label='Gaussian $\cdotp$ $\gamma$={}'.format(gamma) if v == 0 else 't-Student({}) $\cdotp$ $\gamma$={}'.format(v, gamma))
    axes[0, 0].set_xlabel('{} distribution scale'.format(distribution))
    axes[0, 0].set_ylabel('Absolute entropy error')

    axes[0, 1].plot(data[0], data[4], 'o', alpha=0.25, markersize=3,color=color)
    axes[0, 1].set_xlabel('{} distribution scale'.format(distribution))
    axes[0, 1].set_ylabel('Relative entropy error [\\%]')

    #  + np.sign(v)
    axes[0, 2].plot(data[1], data[2], '.', alpha=0.5, markersize=5, color=color)
    axes[0, 2].plot([0, 5], [0, 5], ':')
    axes[0, 2].set_xlim([-0.05, 1.05*max(data[1])])
    axes[0, 2].set_ylim([-0.05, 1.05*max(data[1])])
    axes[0, 2].set_xlabel('Real entropy')
    axes[0, 2].set_ylabel('Soft estimate')
    axes[0, 2].set_title('Synthetic data (Laplacian dist)')
    # 'Gaussian' if v == 0 else 't-Student({})'.format(v)

# axes[0, 0].legend(['Gaussian', 't-Student({})'.format(gamma)])

axes[0, 0].legend()

for kernel, color in zip(['Gaussian', 't-Student'], ['r', 'g']):
    axes[0, 3].plot(results[kernel][0], results[kernel][1], '.', alpha=0.25, markersize=5, label=kernel, color=color)
axes[0, 3].plot([0, 5], [0, 5], ':')
axes[0, 3].set_xlim([0, 1.05*max(results[kernel][1])])
axes[0, 3].set_ylim([0, 1.05*max(results[kernel][1])])
axes[0, 3].set_xlabel('Real entropy')
axes[0, 3].set_ylabel('Soft estimate')
axes[0, 3].set_title('Latent space of 128$\\times$128 px images')

# fig.suptitle('Random sample: {} dist.; Kernel: {}'.format(distribution, 'Gaussian / t-Student'))

plt.tight_layout()
fig.savefig('fig_entropy_errors.pdf')