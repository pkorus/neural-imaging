#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

# Own libraries and modules
from helpers import dataset, coreutils
from models import compression

from training.compression import train_dcn

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
    parser.add_argument('--param_list', dest='dcn_param_list', default=None, help='JSON file with a dictionary of DCN configurations')    
    
    # General
    parser.add_argument('--out', dest='out_dir', action='store', default='./data/raw/compression/',
                        help='output directory for storing trained models')
    parser.add_argument('--epochs', dest='epochs', action='store', default=1500, type=int,
                        help='maximum number of training epochs')
    parser.add_argument('--lr', dest='learning_rate', action='store', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--v_train', dest='validation_is_training', action='store_true', default=True,
                        help='Use the model in training mode while testing')
    parser.add_argument('--no_flip', dest='noflip', action='store_true', default=False,
                        help='disable flipping (data augmentation)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Resume training from last checkpoint, if possible')

    args = parser.parse_args()

    if not args.dcn:
        print('A DCN needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    parameter_list = []

    try:
        if args.dcn_params is not None:
            parameter_list.append(json.loads(args.dcn_params.replace('\'', '"')))

        if args.dcn_param_list is not None:
            with open(args.dcn_param_list) as f:
                parameter_list.extend(json.load(f))

    except json.decoder.JSONDecodeError as e:
        print('WARNING', 'JSON parsing error: ', e)
        sys.exit(2)
    
    try:
        if args.dcn_params is not None:
            args.dcn_params = json.loads(args.dcn_params.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.dcn_params.replace('\'', '"'))
        sys.exit(2)

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
        'sampling_rate': 100,
        'convergence_threshold': 1e-4,
        'current_epoch': 0,
        'validation_is_training': args.validation_is_training,
        'augmentation_probs': {
            'resize': 0.0,
            'flip_h': 0.0 if args.noflip else 0.5,
            'flip_v': 0.0 if args.noflip else 0.5
        }
    }

    if len(parameter_list) == 0:
        parameter_list.append({})

    print('DCN model: {}'.format(args.dcn))
    print('# DCN parameter list [{}]:'.format(len(parameter_list)))
    for i, params in enumerate(parameter_list):
        print('  {:3d} -> {}'.format(i, params))
        
    print('Training Spec:')
    for key, value in training_spec.items():
        print('  {} -> {}'.format(key, value))

    # Load the dataset
    np.random.seed(training_spec['seed'])    
    data = dataset.IPDataset(args.data, n_images=training_spec['n_images'], v_images=training_spec['v_images'], load='y',
                             val_rgb_patch_size=training_spec['patch_size'], val_n_patches=training_spec['valid_patches'])
    
    for key in ['Training', 'Validation']:
        print('{:>16s} [{:5.1f} GB] : Y -> {} '.format(
            '{} data'.format(key),
            coreutils.mem(data[key.lower()]['y']),
            data[key.lower()]['y'].shape
        ), flush=True)

    for params in parameter_list:

        # Create TF session and graph
        graph = tf.Graph()
        sess = tf.Session(graph=graph)

        # Create a DCN according to the spec
        dcn = getattr(compression, args.dcn)(sess, graph, None, patch_size=training_spec['patch_size'], **params)
        train_dcn({'dcn': dcn}, training_spec, data, args.out_dir)

        # Cleanup
        sess.close()
        del graph


if __name__ == "__main__":
    main()
