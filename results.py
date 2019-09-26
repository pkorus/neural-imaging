#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers import coreutils, results_data

supported_plots = ['accuracy', 'scatter-psnr', 'scatter-ssim', 'progress', 'conf', 'conf-tex', 'ssim', 'psnr', 'df', 'auto']


def save_df(df, dirname, df_filename):
    if args.df is not None:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        df.to_csv(os.path.join(dirname, df_filename), index=False)
        print('> saving dataframe to {}'.format(df_filename))


def display_results(args):

    sns.set('paper', font_scale=1, style="ticks")
    plot = coreutils.match_option(args.plot, supported_plots)

    print('Matched plotting command: {}'.format(plot))

    postfix = [
        coreutils.splitall(args.dir)[-1],
        ','.join(args.nips) if args.nips is not None else None,
        ','.join(args.cameras) if args.cameras is not None else None,
    ]
    postfix = '-'.join(x for x in postfix if x is not None)
    
    if plot in ['ssim', 'psnr', 'accuracy']:

        df = results_data.manipulation_metrics(args.nips, args.cameras, root_dir=args.dir)
        sns.catplot(x='ln', y=plot, col='camera', row='nip', data=df, kind='box')
        save_df(df, args.df, 'manipulation_metrics-{}.csv'.format(postfix))
        plt.show()
        return

    if plot == 'scatter-psnr' or plot == 'scatter-ssim':

        df = results_data.manipulation_metrics(args.nips, args.cameras, root_dir=args.dir)
        print(df)
        g = sns.relplot(x=plot.split('-')[-1], y='accuracy', hue='ln', col='camera', row='nip', data=df,
                    palette=sns.color_palette("Set2", len(df['ln'].unique())))
        save_df(df, args.df, 'manipulation_metrics-{}.csv'.format(postfix))
        plt.show()
        return

    if plot == 'progress':

        cases = []

        if args.cameras is None:
            args.cameras = coreutils.listdir(args.dir, '.', dirs_only=True)
        
        for cam in args.cameras:

            nip_models = args.nips or coreutils.listdir(os.path.join(args.dir, cam), '.', dirs_only=True)

            for nip in nip_models:

                reg_path = os.path.join(args.dir, cam, nip)

                if args.regularization:
                    # If given, use specified regularization strengths
                    reg_list = args.regularization
                else:
                    # Otherwise, auto-detect available scenarios
                    reg_list = coreutils.listdir(reg_path, '.*', dirs_only=True)

                    if len(reg_list) > 4:
                        indices = np.linspace(0, len(reg_list)-1, 4).astype(np.int32)
                        reg_list = [reg_list[i] for i in indices]
                        print('! warning - too many experiments to show - sampling: {}'.format(reg_list))

                for reg in reg_list:
                    for r in coreutils.listdir(os.path.join(reg_path, reg), '[0-9]+', dirs_only=True):
                        print('* found scenario {}'.format((cam, nip, reg, int(r))))
                        cases.append((cam, nip, reg, int(r)))
            
        df, labels = results_data.manipulation_progress(cases, root_dir=args.dir)
        save_df(df, args.df, 'progress-{}.csv'.format(postfix))

        for col in ['psnr', 'accuracy']:
            sns.relplot(x="step", y=col, hue='exp', row='nip', col='camera', style='exp', kind="line", legend="full", aspect=2, height=3, data=df)

        plt.show()
        return
    
    if plot == 'conf' or plot == 'conf-tex':
        
        if isinstance(args.nips, list):
            if len(args.nips) > 1:
                print('WARNING Only one NIP will be used for this plot!')
            args.nips = args.nips[0]
        
        conf = results_data.confusion_data(args.run, root_dir=args.dir)

        if len(conf) == 0:
            print('ERROR No results found!')
            return

        tex_output = plot == 'conf-tex'
        plot_data = not tex_output if len(conf.keys()) < 20 else False

        if plot_data:
            images_x = np.ceil(np.sqrt(len(conf)))
            images_y = np.ceil(len(conf) / images_x)
            f_size = 3
            fig = plt.figure(figsize=(images_x*f_size, images_y*f_size))
                
        for i, (k, c) in enumerate(conf.items()):
            data = (100*c['data']).round(0)
            labels = c['labels']
            if tex_output:
                print(results_data.confusion_to_text(data, labels, k, 'tex'))
            else:
                print(results_data.confusion_to_text(data, labels, k, 'txt'))

            if plot_data:
                acc = np.mean(np.diag(data))
                ax = fig.add_subplot(images_y, images_x, i+1)
                sns.heatmap(data, annot=True, fmt=".0f", linewidths=.5, xticklabels=[x[0] for x in labels], yticklabels=labels)
                ax.set_title('{} : acc={:.1f}'.format(k, acc))

        if plot_data:
            plt.tight_layout()
            plt.show()

        return

    if plot == 'df':

        print('Searching for "training.json" in', args.dir)
        df = results_data.manipulation_summary(args.dir)

        if len(df) > 0:
            if False:
                print(df.groupby('scenario').mean().to_string())
            else:
                gb = df.groupby('scenario')
                counts = gb.size().to_frame(name='reps')
                print(counts.join(gb.agg('mean')).reset_index().to_string())

        save_df(df, args.df, 'summary-{}.csv'.format(postfix))

        return

    if plot == 'auto':

        print('Searching for "training.json" in', args.dir)
        df = results_data.manipulation_summary(args.dir)
        df = df.sort_values('scenario')

        guessed_names = {}

        # Guess scenario
        components = df['scenario'].str.split("/", expand=True)
        for i in components:
            # Try to guess the column name based on content
            template = 'scenario:{}'.format(i)
            if components.iloc[0, i].endswith('Net'):
                guessed_names[template] = 'nip'
            elif components.iloc[0, i].startswith('ln-'):
                guessed_names[template] = 'nip reg.'
            elif components.iloc[0, i].startswith('lc-'):
                guessed_names[template] = 'dcn reg.'
            elif set(components.iloc[:, i].unique()) == {'4k', '8k', '16k'}:
                guessed_names[template] = 'dcn'
            elif all([re.match('^[0-9]{2,3}$', x) for x in components.iloc[:, i].unique()]):
                guessed_names[template] = 'jpeg'
            else:
                guessed_names[template] = template

            df[guessed_names[template]] = components[i]

        df['scenario'] = coreutils.remove_commons(df['scenario'])

        mapping = {}
        mapping_targets = ['col', 'col', 'hue', 'style', 'size']
        mapping_id = 0

        # Choose the feature with most unique values as x axis
        uniques = [len(df[guessed_names['scenario:{}'.format(i)]].unique()) for i in components]

        x_feature = np.argmax(uniques)

        for i in components:
            if i == x_feature:
                continue

            if len(df[guessed_names['scenario:{}'.format(i)]].unique()) > 1:
                mapping[mapping_targets[mapping_id]] = guessed_names['scenario:{}'.format(i)]
                mapping_id += 1

        sns.catplot(x=guessed_names['scenario:{}'.format(x_feature)], y='accuracy', data=df, kind='box', **mapping)
        # sns.catplot(x='scenario:0', y='dcn_ssim', data=df, kind='box', **mapping)
        # sns.scatterplot(x='dcn_ssim', y='accuracy', data=df)
        plt.show()

        if len(df) > 0:
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
    parser.add_argument('--run', dest='run', action='store', default=None, type=int,
                        help='select experiment instance number')
    parser.add_argument('--dir', dest='dir', action='store',
                        default=os.path.join(results_data.ROOT_DIRNAME, 'cvpr2019'),
                        help='Root directory with the results')
    parser.add_argument('--df', dest='df', action='store',
                        default=None,
                        help='Path of the output directory for data frames with results')
    args = parser.parse_args()

    if '/' not in args.dir:
        args.dir = os.path.join(results_data.ROOT_DIRNAME, args.dir)

    display_results(args)
