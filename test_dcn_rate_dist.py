#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:56:32 2019

@author: pkorus
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# For curve fitting and regression
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import uncertainties as unc

# %%
from matplotlib import rc

sns.set('paper', font_scale=2, style="ticks")
sns.set_context("paper")

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

rc('axes', titlesize=14)
rc('axes', labelsize=14)
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('legend', fontsize=10)
rc('figure', titlesize=14)

# %%

plots = OrderedDict()
plots['jpg'] = ('jpeg.csv', {})
plots['jp2'] = ('jpeg2000.csv', {})
plots['bpg'] = ('bpg.csv', {})
plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})


dirname = './data/rgb/kodak512/'
metric = 'ssim'
plot_type = 'averages'

# %%

# Load all needed tables and setup legend labels
labels = []
df_all = []

if isinstance(plots, list):
    for filename, selectors in plots:
        labels.append(os.path.splitext(filename)[0])
        df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
        for k, v in selectors.items():
            if isinstance(v, str) and '*' in v:
                df = df[df[k].str.match(v)]
            else:
                df = df[df[k] == v]
        if len(df) == 0:
            raise(ValueError('No rows matched for column {}'.format(k)))
        df_all.append(df)

elif isinstance(plots, dict):
    for key, (filename, selectors) in plots.items():
        labels.append(key)
        df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
        for k, v in selectors.items():
            if isinstance(v, str) and '*' in v:
                df = df[df[k].str.match(v)]
            else:
                df = df[df[k] == v]
        if len(df) == 0:
            raise(ValueError('No rows matched for column {}'.format(k)))
        df_all.append(df)
else:
    raise ValueError('Unsupported plot definition!')

plot_images = [-1] + df_all[0].image_id.unique().tolist()

# %%

# plot_images = [11, 12]

images_x = int(np.ceil(np.sqrt(len(plot_images))))
images_y = int(np.ceil(len(plot_images) / images_x))
    
fig, ax = plt.subplots(images_y, images_x, sharex=True, sharey=True)
fig.set_size_inches((images_x * 6, images_y * 4))

metric = 'msssim_db'
plot = 'ensemble'

baseline_count = 3
add_legend = True
update_ylim = False
draw_markers = True
db_scale = False
marker_legend = False

# Define a parametric model for the trade-off curve
if metric in {'ssim', 'msssim'}:
    fit_bounds = ([0.1, 1e-5, -1, 0], [3, 10, 7, 0.1])

    def func(x, a, b, c, d):
        return 1/(1 + np.exp(- b * x ** a + c)) - d
else:
    # fit_bounds = ([1e-4, 1e-5, 1e-2, -200], [50, 100, 2, 200])
    # fit_bounds = ([1e-4, 1e-5, 1e-2, -200], [50, 100, 2, 200])

    def func(x, a, b, c, d):
        return a * (x + b) ** c + d
        # return a * np.log(np.clip(b*x**c + d, a_min=1e-9, a_max=1e9))
        # return a / (1 + np.exp(- b * x ** c + d))
        # return a + b * x + c * x ** 2 + d * x **3
        # return a * np.log(b * x ** c + d)

if metric == 'psnr':
    ssim_min = 25
    ssim_max = 45
    metric_label = 'PSNR [dB]'

elif metric == 'msssim_db':
    ssim_min = 10
    ssim_max = 32
    metric_label = 'MS-SSIM [dB]'

elif metric == 'ssim':
    ssim_min = 0.85
    ssim_max = 1
    metric_label = 'SSIM'

elif metric == 'msssim':
    ssim_min = 0.9
    ssim_max = 1
    metric_label = 'MS-SSIM'
else:
    raise ValueError('Unsupported metric!')

# Setup drawing styles
styles = [['r-', 'rx'], ['b--', 'b+'], ['k:', 'k2'], ['g-', 'gx'], ['m-', 'gx'], ['m--', 'gx'], ['m-.', 'gx'], ['m:', 'gx']]
avg_markers = ['', '', '', 'o', 'o', '2', '+', 'x', '^', '.']

# To retain consistent styles across plots, adjust the lists based on the number of baseline methods
if baseline_count < 3:
    styles = styles[(3 - baseline_count):]
    avg_markers = avg_markers[(3 - baseline_count):]

mse_labels = {}

for image_id in plot_images:
    
    title = 'Hey'

    if images_y > 1:
        axes = ax[image_id // images_y, image_id % images_x]
    else:
        axes = ax[image_id % images_x]

    # Select measurements for specific images, if specified
    for dfc in df_all:
        if image_id >= 0:
            dfc['selected'] = dfc['image_id'].apply(lambda x: x == image_id)
        else:
            dfc['selected'] = True

    for index, dfc in enumerate(df_all):
        
    
        bpps = dfc.loc[dfc['selected'], 'bpp'].values
        ssims = dfc.loc[dfc['selected'], metric].values

        # if dump_df:
        #     print('{} matched {} rows -> {}'.format(labels[index], len(dfc.loc[dfc['selected']]), 'debug-{}.csv'.format(labels[index])))
        #     dfc.loc[dfc['selected']].to_csv('debug-{}.csv'.format(labels[index]))

        x = np.linspace(max([0, bpps.min() * 0.5]), min([5, bpps.max() * 2]), 256)

        if plot == 'fit':
            # Fit all of the data to a single curve

            popt, pcov = curve_fit(func, bpps, ssims, bounds=fit_bounds, maxfev=10000)
            axes.plot(x, func(x, *popt), styles[index][0], label=labels[index] if add_legend else None)
            # print(labels[index], *popt)

            # If plotting many images, add confidence intervals
            if len(images) > 5 or len(images) == 0:
                a, b, c, d = unc.correlated_values(popt, pcov)
                if metric in {'ssim', 'msssim'}:
                    py = 1 / (1 + unp.exp(- b * x ** a + c)) - d
                else:
                    py = a * unp.log(np.clip(b * x ** c + d, a_min=1e-9, a_max=1e9))

                nom = unp.nominal_values(py)
                std = unp.std_devs(py)

                axes.plot(x, nom - 1.96 * std, c=styles[index][0][0], alpha=0.2)
                axes.plot(x, nom + 1.96 * std, c=styles[index][0][0], alpha=0.2)
                axes.fill(np.concatenate([x, x[::-1]]), np.concatenate([nom - 1.96 * std, (nom + 1.96 * std)[::-1]]),
                          alpha=0.1, fc=styles[index][0][0], ec='None')

            ssim_min = min([ssim_min, func(x[0], *popt)]) if update_ylim else ssim_min

        elif plot == 'ensemble':
            # Fit individual images to a curve, then average the curves

            if image_id >= 0:
                images = [image_id]
            else:
                images = dfc.image_id.unique()
            
            Y = np.zeros((len(images), len(x)))
            params_av = np.zeros((len(images), 4))
            mse_l = []            

            for image_no, imid in enumerate(images):

                bpps = dfc.loc[dfc['selected'] & (dfc['image_id'] == imid), 'bpp'].values
                ssims = dfc.loc[dfc['selected'] & (dfc['image_id'] == imid), metric].values

                try:
                    popt, pcov = curve_fit(func, bpps, ssims,
                                       bounds=fit_bounds,
                                       # method='lm',
                                       maxfev=10000)
                    ssims_est = func(bpps, *popt)
                    mse = np.mean(np.power(ssims - ssims_est, 2))
                    mse_l.append(mse)
                    # print('MSE for {}:{} = {:.2f}'.format(labels[index], image_no, mse))
                    if mse > 10:
                        print('WARNING Large MSE for {}:{} = {:.2f}'.format(labels[index], image_no, mse))
                        print('  bounds: ', fit_bounds)
                        print('  params: ', popt)
                        # print('  x = {}'.format(list(bpps)))
                        # print('  y = {}'.format(list(ssims)))

                except RuntimeError:
                    print('ERROR', labels[index], 'image =', image_id, 'bpp =', bpps, 'ssims =', ssims)

                params_av[image_no] = popt
                Y[image_no] = func(x, *popt)
                # out_of_range_mask = (x < 0.33 * np.min(bpps)) + (x > np.max(bpps) * 3)
                # Y[image_no, out_of_range_mask] = np.nan

            print('Av MSE for {} = {:.2f}'.format(labels[index], np.mean(mse_l)))
            # if image_id < 0:
            #     print(params_av.round(2))
            mse_labels[labels[index]] = np.mean(mse_l)

            # print(Y.tolist())
            y = np.nanmean(Y, axis=0)
            if db_scale:
                y = -10*np.log10(1 - y)
            axes.plot(x, y, styles[index][0], label='{} ({:.3f})'.format(labels[index], mse_labels[labels[index]]) if add_legend else None)
            ssim_min = min([ssim_min, min(y)]) if update_ylim else ssim_min

        elif plot == 'line':
            # Simple linear interpolation

            axes.plot(bpps, ssims, styles[index][0], label=labels[index] if add_legend else None)
            ssim_min = min([ssim_min, min(ssims)]) if update_ylim else ssim_min

        elif plot == 'averages':
            # For each quality level (QF, #channels) find the average quality level

            dfa = dfc.loc[dfc['selected']]

            if 'n_features' in dfa:
                dfg = dfa.groupby('n_features')
            else:
                dfg = dfa.groupby('quality')

            bpps = dfg.mean()['bpp'].values
            ssims = dfg.mean()[metric].values

            axes.plot(bpps, ssims, styles[index][0], label=labels[index] if add_legend else None, marker=avg_markers[index], alpha=0.65)
            ssim_min = min([ssim_min, min(ssims)]) if update_ylim else ssim_min

        elif plot == 'none':
            pass

        else:
            raise ValueError('Unsupported plot type!')

        if draw_markers:

            if 'entropy_reg' in dfc:

                # No need to draw legend if multiple DCNs are plotted
                detailed_legend = 'full' if marker_legend and index == baseline_count else False

                style_mapping = {}

                if 'n_features' in dfc and len(dfc['n_features'].unique()) > 1:
                    style_mapping['hue'] = 'n_features'

                if 'entropy_reg' in dfc and len(dfc['entropy_reg'].unique()) > 1:
                    style_mapping['size'] = 'entropy_reg'

                if 'quantization' in dfc and len(dfc['quantization'].unique()) > 1:
                    style_mapping['style'] = 'quantization'

                if db_scale:
                    dfc[metric] = -10 * np.log10(1 - dfc[metric])

                g = sns.scatterplot(data=dfc[dfc['selected']], x='bpp', y=metric,
                                palette="Set2", ax=axes, legend=detailed_legend,
                                **style_mapping)

            else:
                if db_scale:
                    ssims = -10 * np.log10(1 - ssims)
                axes.plot(bpps, ssims, styles[index][1], alpha=10 / (sum(dfc['selected'])))

    n_images = len(dfc.loc[dfc['selected'], 'image_id'].unique())
    
    # title = str(mse_labels)

    title = '{} : {}'.format(
        title if title is not None else os.path.split(dirname)[-1],
        '{} images'.format(n_images) if n_images > 1 else dfc.loc[dfc['selected'], 'filename'].unique()[0].replace('.png', '')
    )

    # Fixes problems with rendering using the LaTeX backend
    if add_legend:
        for t in axes.legend().texts:
            t.set_text(t.get_text().replace('_', '-'))

    axes.set_xlim([-0.1, 5.1])
    axes.set_ylim([ssim_min * 0.95, ssim_max])
    # axes.set_ylim([0.90, 1])
    # axes.legend(loc='lower right')
    axes.set_title(title)
    axes.set_xlabel('Effective bpp')
    axes.set_ylabel(metric_label)

    
    

# %% Gaussian Process

import tensorflow as tf
import tensorflow_probability as tfp

class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')
    
    self._length_scale = self.add_variable(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.positive_semidefinite_kernels.ExponentiatedQuadratic(
      amplitude=tf.nn.softplus(0.1 * self._amplitude),
      length_scale=tf.nn.softplus(5. * self._length_scale)
    )

# %%
    
x_range = (0.1, 5)
x = np.array([3.980621337890625, 2.659942626953125, 2.057403564453125, 1.695281982421875, 1.450347900390625, 1.29449462890625, 1.160369873046875, 1.06097412109375, 0.9765625, 0.908660888671875, 0.844970703125, 0.772979736328125, 0.710906982421875, 0.6392822265625, 0.56817626953125, 0.49176025390625, 0.416534423828125, 0.342376708984375])
y = np.array([25.353040129172722, 22.507663029772445, 20.814015239893447, 19.580536110971146, 18.634651382939484, 17.934758469472147, 17.33375482907945, 16.773667142703882, 16.279876737504733, 15.884194413334784, 15.42909154401019, 14.938510156936864, 14.404744388040054, 13.774138868790985, 13.079329182646534, 12.098602658978974, 10.95868627458671, 9.555483795310863])
x_tst = np.linspace(0.1, 5.1)

plt.plot(x, y, 'o-')

num_inducing_points = 40
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=[1], dtype=x.dtype),
    tf.keras.layers.Dense(1, kernel_initializer='ones', use_bias=False),
    tfp.layers.VariationalGaussianProcess(
        num_inducing_points=num_inducing_points,
        kernel_provider=RBFKernelFn(dtype=x.dtype),
        event_shape=[1],
        inducing_index_points_initializer=tf.constant_initializer(
            np.linspace(*x_range, num=num_inducing_points,
                        dtype=x.dtype)[..., np.newaxis]),
        unconstrained_observation_noise_variance_initializer=(
            tf.constant_initializer(
                np.log(np.expm1(1.)).astype(x.dtype))),
    ),
])

# Do inference.
batch_size = 32
loss = lambda y, rv_y: rv_y.variational_loss(
    y, kl_weight=np.array(batch_size, x.dtype) / x.shape[0])
model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), loss=loss)
model.fit(x, y, batch_size=batch_size, epochs=1000, verbose=False)

# Make predictions.
yhats = model(x_tst.reshape(-1, 1)).sample()
with tf.Session():
    yy = yhats.eval()

plt.plot(x, y)
plt.plot(x_tst, yy)
