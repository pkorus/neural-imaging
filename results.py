#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import coreutils, results_data

supported_plots = ['boxplot', 'scatter-psnr', 'scatter-ssim', 'progressplot', 'confusion', 'ssim', 'psnr', 'df']


def display_results(args):
    
    sns.set()
    plot = coreutils.match_option(args.plot, supported_plots)
    
    if plot == 'boxplot':

        for nip in args.nips:
            df = results_data.boxplot_data(nip, args.cameras, root_dir=args.dir)
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
            df = results_data.boxplot_data(nip, args.cameras, field=plot, root_dir=args.dir)
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
            df = results_data.scatterplot_data(args.nips, cam, root_dir=args.dir)
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
            
        df, labels = results_data.progressplot_data(cases, root_dir=args.dir)

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
        
        conf = results_data.confusion_data(args.nips, args.cameras, root_dir=args.dir)

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
        df = results_data.manipulation_summary(args)

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
                        default=os.path.join(results_data.ROOT_DIRNAME, 'cvpr2019'),
                        help='Root directory with the results')
    parser.add_argument('--df', dest='df', action='store',
                        default=None,
                        help='Path of the output directory for data frames with results')
    args = parser.parse_args()

    if '/' not in args.dir:
        args.dir = os.path.join(results_data.ROOT_DIRNAME, args.dir)

    if args.nips is None:
        args.nips = ['UNet', 'DNet', 'INet']

    display_results(args)
