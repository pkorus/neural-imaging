#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import glob
import argparse
from collections import namedtuple, OrderedDict
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

from helpers import coreutils, results_data

supported_out = ['raw', 'tex', 'plot']


def main(args):
    
    output = coreutils.match_option(args.output, supported_out)

    if isinstance(args.nips, list):
        if len(args.nips) > 1:
            print('WARNING Only one NIP will be used for this plot!')
        args.nips = args.nips[0]
    
    conf = results_data.confusion_data(args.nips, args.cameras, args.run, args.reg, root_dir=args.dir)

    if output == 'plot':
        import seaborn as sns
        images_x = np.ceil(np.sqrt(len(conf)))
        images_y = np.ceil(len(conf) / images_x)
        f_size = 3
        sns.set()
        fig = plt.figure(figsize=(images_x*f_size, images_y*f_size))
                
        for i, (k, c) in enumerate(conf.items()):
            data = (100*c['data']).round(0)
            labels = c['labels']
            acc = np.mean(np.diag(data))
            ax = fig.add_subplot(images_y, images_x, i+1)
            sns.heatmap(data, annot=True, fmt=".0f", linewidths=.5, xticklabels=[x[0] for x in labels], yticklabels=labels)
            ax.set_title('{} : acc={}'.format(k, acc))
                    
        plt.tight_layout()
        plt.show()
        sys.exit(0)
    
    if output == 'raw':
        for i, (k, c) in enumerate(conf.items()):
            data = (100*c['data']).round(0)
            labels = c['labels']
            print(results_data.confusion_to_text(data, labels, k, 'txt'))
            
        sys.exit(0)

    if output == 'tex':
        for i, (k, c) in enumerate(conf.items()):
            data = (100*c['data']).round(0)
            labels = c['labels']
            print(results_data.confusion_to_text(data, labels, k, 'tex'))
            
        sys.exit(0)

    print('No output mode matched!')


if coreutils.is_interactive():
    Args = namedtuple('Args', 'plot,nips,cameras')
    args = Args('box', ['UNet'], ['Nikon D90'])
    main(args)
            
elif __name__ == "__main__":   
    parser = argparse.ArgumentParser(description='Show results from NIP & FAN optimization')
    parser.add_argument('output', help='Output type (raw, tex, plot)')
    parser.add_argument('--nip', dest='nips', action='append',
                        help='the NIP model (INet, UNet, SigNet)')
    parser.add_argument('--cam', dest='cameras', action='append',
                        help='add cameras for evaluation (repeat if needed)')
    parser.add_argument('--run', dest='run', action='store', default=None, type=int,
                        help='experiment instance number')
    parser.add_argument('--reg', dest='reg', action='store', default=None, type=int,
                        help='regularization index')
    parser.add_argument('--dir', dest='dir', action='store',
                        default='./data/raw/train_manipulation',
                        help='Root directory with the results')
    args = parser.parse_args()

    # Validation
    if args.cameras is None or len(args.cameras) == 0:
        print('No cameras specified!')
        sys.exit(1)

    if args.nips is None:
        print('No NIP specified!')
        sys.exit(1)

    main(args)
