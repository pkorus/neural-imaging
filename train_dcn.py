#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

# Own libraries and modules
from helpers import dataset, coreutils, utils
from models import compression

from training.compression import train_dcn
import pandas as pd

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    
    # Parameters related to the training data
    parser.add_argument('--data', dest='data', action='store', default='./data/compression/',
                       help='directory with training & validation images (png)')
    parser.add_argument('--split', dest='split', action='store', default='16000:800:2',
                       help='data split with #training:#validation:#validation_patches - e.g., 16000:800:2')
    parser.add_argument('--patch', dest='patch_size', action='store', default=128, type=int,
                        help='training patch size')
    
    # Parameters of the DCN
    parser.add_argument('--dcn', dest='dcn', action='store', help='specific DCN class name')
    parser.add_argument('--params', dest='dcn_params', action='append', help='Extra parameters for DCN constructor (JSON string)')
    parser.add_argument('--param_list', dest='dcn_param_list', default=None, help='CSV file with DCN configurations')
    
    # General
    parser.add_argument('--out', dest='out_dir', action='store', default='./data/raw/compression/',
                        help='output directory for storing trained models')
    parser.add_argument('--epochs', dest='epochs', action='store', default=1500, type=int,
                        help='maximum number of training epochs')
    parser.add_argument('--v_schedule', dest='validation_schedule', action='store', default=100, type=int,
                        help='Validation schedule - evaluate the model every v_schedule epochs')    
    parser.add_argument('--lr', dest='learning_rate', action='store', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--v_train', dest='validation_is_training', action='store_true', default=True,
                        help='Use the model in training mode while testing')
    parser.add_argument('--no_flip', dest='noflip', action='store_true', default=False,
                        help='disable flipping (data augmentation)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Resume training from last checkpoint, if possible')
    parser.add_argument('--dry', dest='dry', action='store_true', default=False,
                        help='Dry run (no training - only does model setup)')

    args = parser.parse_args()

    if not args.dcn:
        print('A DCN needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    parameters = pd.DataFrame(columns=['name', 'active'])

    try:
        if args.dcn_params is not None:

            for p in args.dcn_params:
                cli_params = json.loads(p.replace('\'', '"'))
                cli_params['name'] = 'cli'
                cli_params['active'] = True

                parameters = parameters.append(cli_params, ignore_index=True)

        if args.dcn_param_list is not None:
            parameters = parameters.append(pd.read_csv(args.dcn_param_list), ignore_index=True, sort=True)

    except json.decoder.JSONDecodeError as e:
        print('WARNING', 'JSON parsing error: ', e)
        sys.exit(2)

    # Round the number of epochs to align with the sampling rate
    args.epochs = int(np.ceil(args.epochs / args.validation_schedule) * args.validation_schedule) + 1

    training_spec = {
        'seed': 1234,
        'dataset': args.data,
        'n_images': int(args.split.split(':')[0]),
        'v_images': int(args.split.split(':')[1]),
        'valid_patches': int(args.split.split(':')[2]),
        'n_epochs': args.epochs,
        'batch_size': 40,
        'patch_size': args.patch_size,
        'sample_dropout': False,
        'learning_rate': args.learning_rate,
        'learning_rate_reduction_schedule': 1000,
        'learning_rate_reduction_factor': 0.5,
        'validation_schedule': args.validation_schedule,
        'convergence_threshold': 1e-4,
        'current_epoch': 0,
        'validation_is_training': args.validation_is_training,
        'augmentation_probs': {
            'resize': 0.0,
            'flip_h': 0.0 if args.noflip else 0.5,
            'flip_v': 0.0 if args.noflip else 0.5
        }
    }

    if np.sum(parameters['active'] == True) == 0:
        parameters.append({'name': 'default', 'active': True}, ignore_index=True)

    print('DCN model: {}'.format(args.dcn))

    parameters = parameters[parameters['active']].drop(columns=['active'])
    print('# DCN parameter list [{} active configs]:\n'.format(len(parameters)))
    print(parameters)

    print('\n# Training Spec:')
    for key, value in training_spec.items():
        print(' {:50s}: {}'.format(key, value))

    # Load the dataset
    print('\n# Dataset:')
    np.random.seed(training_spec['seed'])    
    data = dataset.IPDataset(args.data, n_images=training_spec['n_images'], v_images=training_spec['v_images'], load='y',
                             val_rgb_patch_size=training_spec['patch_size'], val_n_patches=training_spec['valid_patches'])
    
    for key in ['Training', 'Validation']:
        print('{:>16s} [{:5.1f} GB] : Y -> {} '.format(
            '{} data'.format(key),
            coreutils.mem(data[key.lower()]['y']),
            data[key.lower()]['y'].shape
        ), flush=True)

    model_log = {}

    print('\n# Training:\n')
    for index, params in parameters.drop(columns=['name']).iterrows():

        print('## Scenario {} / {}'.format(index, len(parameters)))
        # Create TF session and graph
        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        # Create a DCN according to the spec
        dcn_params = {k: v for k, v in params.to_dict().items() if not utils.is_nan(v)}
        dcn_params['default_val_is_train'] = training_spec['validation_is_training']
        dcn = getattr(compression, args.dcn)(sess, graph, None, patch_size=training_spec['patch_size'], **dcn_params)

        model_code = dcn.model_code

        if model_code in model_log:
            print('WARNING - model {} already registered by scenario {}'.format(model_code, index))
            model_log[model_code].append(index)
        else:
            model_log[model_code] = [index]

        if not args.dry:
            train_dcn({'dcn': dcn}, training_spec, data, args.out_dir)

        # Cleanup
        sess.close()
        del graph

    if args.dry:
        print('List of instantiated models [{}]:'.format(len(model_log)))
        for index, key in enumerate(sorted(model_log.keys())):
            print('{}  {:3d}. {} -> {}'.format(' ' if len(model_log[key]) == 1 else '!', index, key, model_log[key]))


if __name__ == "__main__":
    main()
