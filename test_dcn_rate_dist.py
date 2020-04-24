#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict

# Toolbox imports
from helpers import utils, plots, tf_helpers
from compression.ratedistortion import plot_bulk

utils.setup_logging()
plots.configure('tex')
tf_helpers.disable_warnings()


def main():
    parser = argparse.ArgumentParser(description='Compare rate-distortion profiles for various codecs')
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
    parser.add_argument('-x', '--markers', dest='markers', action='store', default=1, type=int,
                        help='Draw markers: 0 (none), 1 (only single images), 2 (all markers for the dcn aggregate)')
    
    args = parser.parse_args()
    args.codec = args.codec.split(',')
    args.images = [int(x) for x in args.images]

    if args.data.endswith('/') or args.data.endswith('\\'):
        args.data = args.data[:-1]

    plots = OrderedDict()
    if 'jpg' in args.codec: plots['jpg'] = ('jpeg.csv', {})
    if 'jp2' in args.codec: plots['jp2'] = ('jpeg2000.csv', {})
    if 'bpg' in args.codec: plots['bpg'] = ('bpg.csv', {})
    if 'dcn' in args.codec: plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})
    
    baseline_count = sum([x in args.codec for x in ['jpg', 'jp2', 'bpg']])

    fig = plot_bulk(plots, args.data, args.images, args.metric, args.plot, baseline_count, True, args.max_bpp, args.markers)

    # Save or display
    if args.output is not None:
        plt.tight_layout(3)
        dset = os.path.split(args.data)[-1]
        of_name = os.path.join(args.output, 'tradeoff_{}_{}_{}.pdf'.format(dset, args.metric, args.plot))
        fig.savefig(of_name, bbox_inches='tight')
        print('Wrritten to {}'.format(of_name))
    else:
        plt.tight_layout(3)
        plt.show()
        plt.close()


if not utils.is_interactive():
    main()

else:
    plots = OrderedDict()
    plots['jpg'] = ('jpeg.csv', {})
    plots['jp2'] = ('jpeg2000.csv', {})
    plots['bpg'] = ('bpg.csv', {})
    plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})
    
    dirname = './data/rgb/raw512'
    metric = 'ssim'
    plot = 'agg'
    
    fig = plot_bulk(plots, dirname, [-1], metric, plot, 3, True, 5, draw_markers=1)
    fig.show()
