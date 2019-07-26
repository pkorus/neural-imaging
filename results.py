#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import json
import glob
from pathlib import Path
import argparse
from collections import OrderedDict
from itertools import product
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import coreutils

supported_plots = ['boxplot', 'scatter-psnr', 'scatter-ssim', 'progressplot', 'confusion', 'ssim', 'psnr', 'df']

ROOT_DIRNAME = './data/raw/m'


def boxplot_data(nip_model, cameras=None, field='accuracy', root_dir=ROOT_DIRNAME):
    
    cameras = cameras or coreutils.listdir(root_dir, '.', dirs_only=True)

    find_dir = os.path.join(root_dir, cameras[0], nip_model)
    experiment_dirs = coreutils.listdir(os.path.join(find_dir), 'lr-.*', dirs_only=True)

    df = pd.DataFrame(columns=['{} ({})'.format(c, e) for c, e in product(cameras, experiment_dirs)])
        
    for camera in cameras:

        for ed in experiment_dirs:
            
            column = '{} ({})'.format(camera, ed)
            
            find_dir = os.path.join(root_dir, camera, nip_model, ed)
            jsons_files = sorted(glob.glob(os.path.join(find_dir, '**', 'training.json')))
            
            accuracies = []
            
            for jf in jsons_files:
                with open(jf) as f:
                    data = json.load(f)
                
                if field == 'accuracy':
                    accuracies.append(data['forensics']['validation']['accuracy'][-1])
                elif field == 'psnr':
                    accuracies.append(data['nip']['validation']['psnr'][-1])
                elif field == 'ssim':
                    accuracies.append(data['nip']['validation']['ssim'][-1])
                
            if len(jsons_files):
                
                for _ in range(len(df) - len(accuracies)):
                    accuracies.append(np.nan)
                    
                df[column] = accuracies
            
            else:
                df = df.drop(columns=column)
    
    return df


def scatterplot_data(nip_models, camera, root_dir=ROOT_DIRNAME):
    
    nip_models = nip_models if type(nip_models) is list else [nip_models]

    if camera is None:
        print('! warning: camera not specified- using first available one!')
    camera = camera or coreutils.listdir(root_dir, '.', dirs_only=True)[0]
    
    df = pd.DataFrame(columns=['camera', 'nip', 'lr', 'source', 'psnr', 'ssim', 'accuracy'])
    
    for nip in nip_models:
    
        find_dir = os.path.join(root_dir, camera, nip)
        experiment_dirs = coreutils.listdir(os.path.join(find_dir), 'lr-.*', dirs_only=True)
            
        for ed in experiment_dirs:
            
            exp_dir = os.path.join(find_dir, ed)
            jsons_files = sorted(glob.glob(os.path.join(exp_dir, '**', 'training.json')))
            
            for jf in jsons_files:
                with open(jf) as f:
                    data = json.load(f)
                
                df = df.append({'camera': camera,
                                'nip': nip,
                                'lr': re.findall('(lr-[0-9]\.[0-9]{4})', jf.replace(find_dir, ''))[0],
                                'source': jf.replace(find_dir, '').replace('training.json', ''),
                                'psnr': data['nip']['validation']['psnr'][-1],
                                'ssim': data['nip']['validation']['ssim'][-1],
                                'accuracy': data['forensics']['validation']['accuracy'][-1]
                    }, ignore_index=True)
    
    return df


def progressplot_data(cases, root_dir=ROOT_DIRNAME):
    
    cases = cases or [('Nikon D90', 'INet', 'lr-0.0000', 0)]

    df = pd.DataFrame(columns=['camera', 'nip', 'exp', 'rep', 'step', 'psnr', 'ssim', 'accuracy'])
    labels = []
    
    l_camera, l_nip_model, l_ed, l_rep = None, None, None, None
    
    for i, (camera, nip_model, ed, rep) in enumerate(cases):
        
        # If something is unspecified, use the last known value
        camera = camera or l_camera
        nip_model = nip_model or l_nip_model
        ed = ed or l_ed
        rep = rep if rep is not None else l_rep                
        
        filename = os.path.join(root_dir, camera, nip_model, ed, '{:03d}'.format(rep), 'training.json')

        if not os.path.isfile(filename):
            print('! warning: could not find file {}'.format(filename))
            continue

        labels.append('{0} ({1}/{2}/{3})'.format(camera, nip_model, ed, rep))            
        
        with open(filename) as f:
            data = json.load(f)

        def match_length(y, x):
            x = x[:len(y)]
            for _ in range(len(y) - len(x)):
                x.append(x[-1])
            return x        
        
        d_psnr = data['nip']['validation']['psnr']
        d_ssim = data['nip']['validation']['ssim']
        d_accuracy = data['forensics']['validation']['accuracy']
                    
        df = df.append({
                'camera': [camera] * len(d_accuracy),
                'nip': [nip_model] * len(d_accuracy),
                'exp': [ed] * len(d_accuracy),
                'rep': [rep] * len(d_accuracy),
                'step': list(range(len(d_accuracy))),
                'psnr': match_length(d_accuracy, d_psnr),
                'ssim': match_length(d_accuracy, d_ssim),
                'accuracy': d_accuracy
                }, ignore_index=True, sort=False)
                
        # Remember last used values for future iterations
        l_camera, l_nip_model, l_ed, l_rep = camera, nip_model, ed, rep

    if len(df) == 0:
        raise RuntimeError('Empty dataframe! Double check experimental scenario!')
            
    return df, labels


def confusion_data(nip_model, cameras, root_dir=ROOT_DIRNAME):
    cameras = cameras or coreutils.listdir(root_dir, '.', dirs_only=True)
    
    confusion = OrderedDict()
    
    for camera in cameras:

        experiment_dirs = sorted(coreutils.listdir(os.path.join(root_dir, camera, nip_model), '.', dirs_only=True))
        
        for ed in experiment_dirs:
            
            find_dir = os.path.join(root_dir, camera, nip_model, ed)
            jsons_files = sorted(glob.glob(os.path.join(find_dir, '**', 'training.json')))            
            
            if len(jsons_files) > 1:           
                print('WARNING Using the first found repetition of the experiment')
            jf = jsons_files[0]
            
            with open(jf) as f:
                data = json.load(f)                                
            confusion['{}/{}'.format(camera, ed)] = {
                    'data': np.array(data['forensics']['validation']['confusion']),
                    'labels': data['summary']['Classes'] if isinstance(data['summary']['Classes'], list) else eval(data['summary']['Classes'])
                    }
                    
    return confusion
    

def display_results(args):
    
    sns.set()
    plot = coreutils.match_option(args.plot, supported_plots)
    
    if plot == 'boxplot':

        for nip in args.nips:
            df = boxplot_data(nip, args.cameras, root_dir=args.dir)
            print(df)
            print('Averages')
            print(df.mean().round(2))
            plt.figure()
            sns.boxplot(data=df)
            plt.xticks(rotation=90)
            plt.gca().set_title(nip)

            if args.df is not None:
                if not os.path.isdir(args.df):
                    os.makedirs(args.df)
                df_filename = '{}/box-{}-{}-{}.csv'.format(args.df, 'accuracy', nip, plot)
                df.to_csv(df_filename, index=False)
                print('> saving dataframe to {}'.format(df_filename))
        plt.show()
        return
        
    if plot == 'psnr' or plot == 'ssim':

        for nip in args.nips:
            df = boxplot_data(nip, args.cameras, field=plot, root_dir=args.dir)
            print(df)
            print('Averages')
            print(df.mean().round(1 if plot == 'psnr' else 3))
            plt.figure()
            sns.boxplot(data=df)
            plt.xticks(rotation=90)
            plt.gca().set_title(nip)

            if args.df is not None:
                if not os.path.isdir(args.df):
                    os.makedirs(args.df)
                df_filename = '{}/box-{}-{}-{}.csv'.format(args.df, plot, nip, plot)
                df.to_csv(df_filename, index=False)
                print('> saving dataframe to {}'.format(df_filename))

        plt.show()
        return

    if plot == 'scatter-psnr' or plot == 'scatter-ssim':

        if args.cameras is None:
            args.cameras = coreutils.listdir(args.dir, '.', dirs_only=True)

        for cam in args.cameras:
            df = scatterplot_data(args.nips, cam, root_dir=args.dir)
            print(df)
            sns.relplot(x=plot.split('-')[-1], y='accuracy', hue='lr', col='camera', data=df)

            if args.df is not None:
                if not os.path.isdir(args.df):
                    os.makedirs(args.df)
                df_filename = '{}/scatter-{}-{}.csv'.format(args.df, cam, ','.join(args.nips))
                df.to_csv(df_filename, index=False)
                print('> saving dataframe to {}'.format(df_filename))
        plt.show()
        return

    if plot == 'progressplot':

        cases = []

        if args.cameras is None:
            args.cameras = coreutils.listdir(args.dir, '.', dirs_only=True)
        
        for cam in args.cameras:
            for nip in args.nips:

                reg_path = os.path.join(args.dir, cam, nip)

                if args.regularization:
                    # If given, use specified regularization strengths
                    reg_list = args.regularization
                else:
                    # Otherwise, auto-detect available scenarios
                    reg_list = coreutils.listdir(reg_path, 'lr-[0-9\.]+', dirs_only=True)

                    if len(reg_list) > 4:
                        indices = np.linspace(0, len(reg_list)-1, 4).astype(np.int32)
                        reg_list = [reg_list[i] for i in indices]
                        print('! warning - too many experiments to show - sampling: {}'.format(reg_list))

                for reg in reg_list:
                    for r in coreutils.listdir(os.path.join(reg_path, reg), '[0-9]+', dirs_only=True):
                        print('* found scenario {}'.format((cam, nip, reg, int(r))))
                        cases.append((cam, nip, reg, int(r)))
            
        df, labels = progressplot_data(cases, root_dir=args.dir)

        if args.df is not None:
            if not os.path.isdir(args.df):
                os.makedirs(args.df)
            df_filename = '{}/progress-{}-{}.csv'.format(args.df, ','.join(args.cameras), ','.join(args.nips))
            df.to_csv(df_filename, index=False)
            print('> saving dataframe to {}'.format(df_filename))

        for col in ['psnr', 'accuracy']:
            sns.relplot(x="step", y=col, hue='exp', col='nip', row='camera', style='exp', kind="line", legend="full", aspect=2, height=3, data=df)
            
        plt.show()
        return
    
    if plot == 'confusion':
        
        if isinstance(args.nips, list):
            if len(args.nips) > 1:
                print('WARNING Only one NIP will be used for this plot!')
            args.nips = args.nips[0]
        
        conf = confusion_data(args.nips, args.cameras, root_dir=args.dir)

        images_x = np.ceil(np.sqrt(len(conf)))
        images_y = np.ceil(len(conf) / images_x)
        f_size = 3
        fig = plt.figure(figsize=(images_x*f_size, images_y*f_size))
                
        for i, (k, c) in enumerate(conf.items()):
            data = (100*c['data']).round(0)
            labels = c['labels']
            print('\n', k, '=')
            print(data)
            print(labels)
            acc = np.mean(np.diag(data))
            print('Accuracy = {}'.format(acc))
            ax = fig.add_subplot(images_y, images_x, i+1)
            sns.heatmap(data, annot=True, fmt=".0f", linewidths=.5, xticklabels=[x[0] for x in labels], yticklabels=labels)
            ax.set_title('{} : acc={}'.format(k, acc))

        plt.tight_layout()
        plt.show()
        return

    if plot == 'df':

        print('Searching for "training.json" in', args.dir)

        df = pd.DataFrame(columns=['scenario', 'run', 'accuracy', 'nip_ssim', 'nip_psnr', 'dcn_ssim', 'dcn_entropy'])

        for filename in Path(args.dir).glob('**/training.json'):
            with open(str(filename)) as f:
                data = json.load(f)

            default = [np.nan]
            accuracy = coreutils.getkey(data, 'forensics/validation/accuracy') or default
            nip_ssim = coreutils.getkey(data, 'nip/validation/ssim') or default
            nip_psnr = coreutils.getkey(data, 'nip/validation/psnr') or default
            dcn_ssim = coreutils.getkey(data, 'compression/validation/ssim') or default
            dcn_entr = coreutils.getkey(data, 'compression/validation/entropy') or default

            path_components = os.path.relpath(str(filename), args.dir).split('/')[:-1]

            df = df.append({
                'scenario': os.path.join(*path_components[:-1]),
                'run': int(path_components[-1]),
                'accuracy': accuracy[-1],
                'nip_ssim': nip_ssim[-1],
                'nip_psnr': nip_psnr[-1],
                'dcn_ssim': dcn_ssim[-1],
                'dcn_entropy': dcn_entr[-1]
            }, ignore_index=True, sort=False)

        if len(df) > 0:
            if False:
                print(df.groupby('scenario').mean().to_string())
            else:
                gb = df.groupby('scenario')
                counts = gb.size().to_frame(name='reps')
                print(counts.join(gb.agg('mean')).reset_index().to_string())

        return

    raise RuntimeError('No plot matched! Available plots {}'.format(', '.join(supported_plots)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Show results from NIP & FAN optimization')
    parser.add_argument('plot', help='Plot type ({})'.format(', '.join(supported_plots)))
    parser.add_argument('--nip', dest='nips', action='append',
                        help='the NIP model (INet, UNet, DNet)')
    parser.add_argument('--cam', dest='cameras', action='append',
                        help='add cameras for evaluation (repeat if needed)')
    parser.add_argument('--r', dest='regularization', action='append',
                        help='add regularization strength (repeat if needed)')
    parser.add_argument('--dir', dest='dir', action='store',
                        default=os.path.join(ROOT_DIRNAME, 'cvpr2019'),
                        help='Root directory with the results')
    parser.add_argument('--df', dest='df', action='store',
                        default=None,
                        help='Path of the output directory for data frames with results')
    args = parser.parse_args()

    if '/' not in args.dir:
        args.dir = os.path.join(ROOT_DIRNAME, args.dir)

    if args.nips is None:
        args.nips = ['UNet', 'DNet', 'INet']

    display_results(args)
