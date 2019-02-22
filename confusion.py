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

from helpers import coreutils

supported_out = ['raw', 'tex', 'plot']

coreutils.logCall
def confusion_data(nip_model, cameras, run=None, reg=None, root_dir='./data/raw/train_manipulation'):    
    cameras = cameras or coreutils.listdir(root_dir, '.', dirs_only=True)    
    confusion = OrderedDict()
    
    for camera in cameras:

        experiment_dirs = sorted(coreutils.listdir(os.path.join(root_dir, camera, nip_model), '.', dirs_only=True))

        if reg is not None:
            experiment_dirs = experiment_dirs[reg:reg+1]
        
        for ed in experiment_dirs:
            
            find_dir = os.path.join(root_dir, camera, nip_model, ed)
            jsons_files = sorted(glob.glob(os.path.join(find_dir, '**', 'training.json')))       

            if run is None:
                print('WARNING Using the first found repetition of the experiment')
                run = 0
            jf = jsons_files[run]
            
            with open(jf) as f:
                data = json.load(f)                                
            confusion['{}/{}'.format(camera, ed)] = {
                    'data': np.array(data['forensics']['validation']['confusion']),
                    'labels': data['summary']['Classes'] if isinstance(data['summary']['Classes'], list) else eval(data['summary']['Classes'])
                    }
                    
    return confusion

def conf2tex(conf, labels, title=''):
    if not isinstance(conf, np.ndarray):
        conf = np.array(conf)

    if conf.ndim != 2:
        raise ValueError('2D array expected!')

    n = conf.shape[0]
    l = max([len(x) for x in labels])

    # Append the pre-amble
    out = []
    out.append('\\documentclass[preview]{standalone}\n')
    out.append('\\usepackage{booktabs}\n')
    out.append('\\usepackage{diagbox}\n')
    out.append('\\usepackage{graphicx}\n')
    out.append('\\usepackage{xcolor,colortbl}\n')
    out.append('\\begin{document}\n')
    out.append('\\begin{preview}\n')
    out.append('\\begin{{tabular}}{{l{0}}}\n'.format(n*'r'))
    out.append('\\multicolumn{{{0}}}{{c}}{{{1} $\\rightarrow$ {2}\\%}} '.format(n+1, title, np.mean(np.diag(conf))))
    out.append('\\tabularnewline\n')
    out.append('\\diagbox{\\textbf{True}}{\\textbf{Predicted}}')

    # Fill the header with class names
    for i in range(n):
        out.append('& \\rotatebox{{90}}{{\\textbf{{{0}}}}}'.format(labels[i]))
    out.append(' \\tabularnewline\n')
    out.append('\\toprule\n')

    for i in range(n):
        out.append('\\textbf{{{0}}} '.format(labels[i]))
        for j in range(n):
            if conf[i][j] == 0:
                out.append('& ')
            elif conf[i][j] < 3:
                out.append('& *')
            else:
                out.append('& \\cellcolor{{{0}!{1:.0f}}} {1:.0f}'.format('lime' if i == j else 'red', conf[i][j]))
        out.append(' \\tabularnewline\n')

    out.append('\\bottomrule\n')
    out.append('\\end{tabular}\n')
    out.append('\\end{preview}\n')
    out.append('\\end{document}\n')
    return ''.join(out)
    

def conf2txt(conf, labels, title=''):
    if not isinstance(conf, np.ndarray):
        conf = np.array(conf)

    if conf.ndim != 2:
        raise ValueError('2D array expected!')

    n = conf.shape[0]
    l = max([len(x) for x in labels])

    out = []

    out.append('# {} (acc={:.1f})'.format(title, np.mean(np.diag(conf))))
    out.append('\n')
    out.append(' '*l)
    for i in range(n):
        out.append('{:>4}'.format(labels[i][0]))
    out.append('\n')
    for i in range(n):
        out.append('{:>{width}}'.format(labels[i], width=l))
        for j in range(n):
            out.append('{:4.0f}'.format(conf[i][j]))
        out.append('\n')
    
    return ''.join(out)

def main(args):
    
    output = coreutils.match_option(args.output, supported_out)

    if isinstance(args.nips, list):
        if len(args.nips) > 1:
            print('WARNING Only one NIP will be used for this plot!')
        args.nips = args.nips[0]
    
    conf = confusion_data(args.nips, args.cameras, args.run, args.reg, root_dir=args.dir)

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
            print(conf2txt(data, labels, k))
            
        sys.exit(0)

    if output == 'tex':
        for i, (k, c) in enumerate(conf.items()):
            data = (100*c['data']).round(0)
            labels = c['labels']
            print(conf2tex(data, labels, k))
            
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
