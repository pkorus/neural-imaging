#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 14:56:32 2019

@author: pkorus
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# For curve fitting and regression
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import uncertainties as unc

# Toolbox imports
from helpers import coreutils

# %% Setup plots
from matplotlib import rc

sns.set('paper', font_scale=2, style="ticks")
sns.set_context("paper")

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

rc('figure', dpi=72)
rc('axes', titlesize=14)
rc('axes', labelsize=14)
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('legend', fontsize=10)
rc('figure', titlesize=14)

# %%

def load_data(plots, dirname):
    """
    Returns dataframes with numerical results for specified codecs [and settings]

    Example definition (can be both a list or a dictionary):

    plots = OrderedDict()
    plots['jpg'] = ('jpeg.csv', {})
    plots['jp2'] = ('jpeg2000.csv', {})
    plots['bpg'] = ('bpg.csv', {})
    plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})

    Tuple structure: (filename, data filtering conditions - dict {column: value})

    """

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
        
    return df_all, labels

def setup_plot(metric):
    
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
    
    return ssim_min, ssim_max, metric_label

def setup_fit(metric):
    
    # Define a parametric model for the trade-off curve
    if metric in {'ssim', 'msssim'}:
        fit_bounds = ([0.1, 1e-5, -1, 0], [3, 10, 7, 0.1])
    
        def func(x, a, b, c, d):
            return 1/(1 + np.exp(- b * x ** a + c)) - d
    else:
        # fit_bounds = ([1e-4, 1e-5, 1e-2, -200], [50, 100, 2, 200])
        fit_bounds = ([1e-4, 1e-5, 1e-2, -50], [100, 100, 3, 50])
    
        def func(x, a, b, c, d):
            # return a * (x + b) ** c + d
            return a * np.log(np.clip(b*x**c + d, a_min=1e-9, a_max=1e9))
            # return a / (1 + np.exp(- b * x ** c + d))
            # return a + b * x + c * x ** 2 + d * x **3
            # return a * np.log(b * x ** c + d)
        
    return func, fit_bounds
        

# %%

def plot_ratedistortion(plots, dirname, plot_images, metric, plot, baseline_count=3, add_legend=True, output=None, max_bpp=5, draw_markers=1):
    
    plot = coreutils.match_option(plot, ['fit', 'aggregate'])
    dirname = dirname.strip('/')
        
    # Load data and select images for plotting    
    df_all, labels = load_data(plots, dirname)
    plot_images = plot_images if len(plot_images) > 0 else [-1] + df_all[0].image_id.unique().tolist()
    print(plot_images)

    images_x = int(np.ceil(np.sqrt(len(plot_images))))
    images_y = int(np.ceil(len(plot_images) / images_x))
                
    update_ylim = False
    db_scale = False
    marker_legend = False

    # Plot setup    
    func, fit_bounds = setup_fit(metric)
    ssim_min, ssim_max, metric_label = setup_plot(metric)
    
    # Setup drawing styles
    styles = [['r-', 'rx'], ['b--', 'b+'], ['k:', 'k2'], ['g-', 'gx'], ['m-', 'gx'], ['m--', 'gx'], ['m-.', 'gx'], ['m:', 'gx']]
    avg_markers = ['', '', '', 'o', 'o', '2', '+', 'x', '^', '.']
    
    # To retain consistent styles across plots, adjust the lists based on the number of baseline methods
    if baseline_count < 3:
        styles = styles[(3 - baseline_count):]
        avg_markers = avg_markers[(3 - baseline_count):]
    
    mse_labels = {}

    fig, ax = plt.subplots(images_y, images_x, sharex=True, sharey=True)
    fig.set_size_inches((images_x * 6, images_y * 4))
        
    for image_id in plot_images:
        
        if images_y > 1:
            axes = ax[image_id // images_x, image_id % images_x]
        elif images_x > 1:
            axes = ax[image_id % images_x]
        else:
            axes = ax
    
        # Select measurements for a specific image, if specified
        for dfc in df_all:
            if image_id >= 0:
                dfc['selected'] = dfc['image_id'].apply(lambda x: x == image_id)
            else:
                dfc['selected'] = True
    
        for index, dfc in enumerate(df_all):
            
            bpps = dfc.loc[dfc['selected'], 'bpp'].values
            ssims = dfc.loc[dfc['selected'], metric].values
        
            x = np.linspace(max([0, bpps.min() * 0.9]), min([5, bpps.max() * 1.1]), 256)
        
            if plot == 'fit':
                # Fit individual images to a curve, then average the curves
    
                if image_id >= 0:
                    images = [image_id]
                else:
                    images = dfc.image_id.unique()
                
                Y = np.zeros((len(images), len(x)))
                mse_l = []            
    
                for image_no, imid in enumerate(images):
    
                    bpps = dfc.loc[dfc['selected'] & (dfc['image_id'] == imid), 'bpp'].values
                    ssims = dfc.loc[dfc['selected'] & (dfc['image_id'] == imid), metric].values
    
                    try:
                        popt, pcov = curve_fit(func, bpps, ssims,
                                           bounds=fit_bounds,
                                           # method='lm',
                                           maxfev=100000)
                        ssims_est = func(bpps, *popt)
                        mse = np.mean(np.power(ssims - ssims_est, 2))
                        mse_l.append(mse)
                        if mse > 1:
                            print('WARNING Large MSE for {} img=#{} = {:.2f}'.format(labels[index], image_no, mse))
                            # print('  bounds: ', fit_bounds)
                            # print('  params: ', popt)
    
                    except RuntimeError as err:
                        print('ERROR', labels[index], 'image =', imid, 'bpp =', bpps, 'ssims =', ssims, 'err =', err)

                    Y[image_no] = func(x, *popt)

                if image_id < 0:
                    print('Fit quality - MSE for {} av={:.2f} max={:.2f}'.format(labels[index], np.mean(mse_l), np.max(mse_l)))
                mse_labels[labels[index]] = np.mean(mse_l)
    
                y = np.nanmean(Y, axis=0)
                axes.plot(x, y, styles[index][0], label='{} ({:.3f})'.format(labels[index], mse_labels[labels[index]]) if add_legend else None)
                ssim_min = min([ssim_min, min(y)]) if update_ylim else ssim_min
    
            elif plot == 'line':
                # Simple linear interpolation
    
                axes.plot(bpps, ssims, styles[index][0], label=labels[index] if add_legend else None)
                ssim_min = min([ssim_min, min(ssims)]) if update_ylim else ssim_min
    
            elif plot == 'aggregate':
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
    
            if draw_markers > 0:
    
                if 'entropy_reg' in dfc:
                    
                    if image_id >= 0 or draw_markers >= 2: 
    
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
                    
                    if image_id >= 0:
                        axes.plot(bpps, ssims, styles[index][1], alpha=0.65)
    
        # Setup title
        n_images = len(dfc.loc[dfc['selected'], 'image_id'].unique())
        if n_images > 1:
            title = '{} for {} images ({})'.format(plot, n_images, os.path.split(dirname)[-1])
        else:
            title = '\#{} : {}'.format(image_id, dfc.loc[dfc['selected'], 'filename'].unique()[0].replace('.png', ''))
    
        # Fixes problems with rendering using the LaTeX backend
        if add_legend:
            for t in axes.legend().texts:
                t.set_text(t.get_text().replace('_', '-'))
    
        axes.set_xlim([-0.1, max_bpp + 0.1])
        axes.set_ylim([ssim_min * 0.95, ssim_max])
        axes.legend(loc='lower right')
        axes.set_title(title)
        if image_id // images_x == images_y - 1:
            axes.set_xlabel('Effective bpp')
        if image_id % images_x == 0:
            axes.set_ylabel(metric_label)
        
    # Save or display
    if output is not None:
        plt.tight_layout(3)
        dset = os.path.split(dirname)[-1]
        of_name = os.path.join(output, 'tradeoff_{}_{}_{}.pdf'.format(dset, metric, plot))
        fig.savefig(of_name, bbox_inches='tight')
        print('Wrritten to {}'.format(of_name))
    else:
        plt.tight_layout(3)
        plt.show()
        plt.close()

    
def main():
    parser = argparse.ArgumentParser(description='Test a neural imaging pipeline')
    parser.add_argument('-d', '--data', dest='data', action='store', default='./data/rgb/clic512',
                        help='directory with training & validation images (png)')
    parser.add_argument('-i', '--images', dest='images', action='append', default=[], 
                        help='select images for plotting')
    parser.add_argument('-m', '--metric', dest='metric', action='store', default='ssim',
                        help='distortion metric (ssim, msssim, msssim_db, psnr')
    parser.add_argument('-p', '--plot', dest='plot', action='store', default='fit',
                        help='plot type (aggregate, fit)')
    parser.add_argument('-c', '--codec', dest='codec', action='store', default='jpg,jp2,bpg,dcn',
                        help='plot type (aggregate, fit)')
    parser.add_argument('-o', '--out', dest='output', action='store', default=None,
                        help='output directory for the figure')
    parser.add_argument('-b', '--bpp', dest='max_bpp', action='store', default=3,
                        help='limit for the rate axis (bpp, default=3)')
    parser.add_argument('-x', '--markers', dest='markers', action='store', default=0, type=int,
                        help='Draw markers: 0 (none), 1 (only single images), 2 (all markers for the dcn aggregate)')
    
    args = parser.parse_args()
    args.codec = args.codec.split(',')
    args.images = [int(x) for x in args.images]
    
    plots = OrderedDict()
    if 'jpg' in args.codec: plots['jpg'] = ('jpeg.csv', {})
    if 'jp2' in args.codec: plots['jp2'] = ('jpeg2000.csv', {})
    if 'bpg' in args.codec: plots['bpg'] = ('bpg.csv', {})
    if 'dcn' in args.codec: plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})
    
    baseline_count = sum([x in args.codec for x in ['jpg', 'jp2', 'bpg']])
            
    plot_ratedistortion(plots, args.data, args.images, args.metric, args.plot, baseline_count, True, args.output, args.max_bpp, args.markers)
    
if not coreutils.is_interactive():
    main()

else:
    plots = OrderedDict()
    plots['jpg'] = ('jpeg.csv', {})
    plots['jp2'] = ('jpeg2000.csv', {})
    plots['bpg'] = ('bpg.csv', {})
    plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})
    
    dirname = './data/rgb/clic512'
    metric = 'ssim'
    plot = 'agg'
    
    plot_ratedistortion(plots, dirname, [], metric, plot, 3, True, None, 5)