#!/usr/bin/env python3
# coding: utf-8
import io
import os
import sys
import json
import tqdm
import imageio
import argparse
import numpy as np
import tensorflow as tf

from collections import deque, OrderedDict
from skimage.transform import resize, rescale
from skimage.measure import compare_ssim as ssim
from scipy.cluster.vq import vq

import matplotlib.pyplot as plt

# Own libraries and modules
from helpers import loading, plotting, utils, summaries, tf_helpers, dataset
from models import compression

from training.compression import train_dcn

# Set progress bar width
TQDM_WIDTH = 120

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
    
    # Genereal
    parser.add_argument('--out', dest='out_dir', action='store', default='./data/raw/compression/',
                        help='output directory for storing trained models')
    parser.add_argument('--epochs', dest='epochs', action='store', default=5000, type=int,
                        help='maximum number of training epochs')
    parser.add_argument('--lr', dest='learning_rate', action='store', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--vtrain', dest='validation_is_training', action='store', default=True, type=bool,
                        help='Use the model in training mode while testing')
    parser.add_argument('--noflip', dest='noflip', action='store_true', default=False, type=bool,
                        help='disable flipping (data augmentation)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Resume training from last checkpoint, if possible')

    args = parser.parse_args()

    if not args.dcn:
        print('A DCN needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    # Lazy loading after parameter sanitization finished
    from helpers import loading

    data_directory = os.path.join(args.data_dir, args.camera)
    out_directory_root = args.out_dir
    
    parameter_list = []

    try:
        if args.dcn_params is not None:
            parameter_list.append(json.loads(args.dcn_params.replace('\'', '"')))

        if args.dcn_param_list is not None:
            with open(dcn_param_list) as f:
                parameter_list.extend(json.load(f))

    except json.decoder.JSONDecodeError as e:
        print('WARNING', 'JSON parsing error: ', e)
        sys.exit(2)
    
    try:
        if args.nip_params is not None:
            args.nip_params = json.loads(args.nip_params.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.nip_params.replace('\'', '"'))
        sys.exit(2)

    training_spec = {
        'seed': 1234,
        'n_images': int(args.split.split(':')[0]),
        'v_images': int(args.split.split(':')[1]),
        'valid_patches': int(args.split.split(':')[2])        
        'n_epochs': args.epochs,
        'batch_size': 40,
        'patch_size': args.patch_size,
        'sample_dropout': False,
        'learning_rate': args.learning_rate,
        'sampling_rate': 100,
        'current_epoch': 0,
        'validation_is_training': args.validation_is_training,
        'augmentation_probs': {
            'resize': 0.0,
            'flip_h': 0.0 if args.noflip else 0.5,
            'flip_v': 0.0 if args.noflip else 0.5
        }
    }    

    print('DCN : {}'.format(args.dcn))    
    print('# DCN parameter list: {}'.format(len(parameter_list)))
    for i, params in enumerate(parameter_list):
        print('  {3d} -> {}'.format(i, params))
        
    print('Training Spec : {}'.format(training_spec))
    
    # Load the dataset
    np.random.seed(training_spec['seed'])    
    data = dataset.IPDataset(args.data, training_spec['n_images'], training_spec['v_images'], load='y', val_patch_size=training['patch_size'])
    
    print('Training data shape (X)   : {}'.format(data['training']['x'].shape))
    print('Training data size        : {:.1f} GB'.format(coreutils.mem(data['training']['x']) + coreutils.mem(data['training']['y'])), flush=True)
    print('Validation data shape (X) : {}'.format(data['validation']['x'].shape))
    print('Validation data size      : {:.1f} GB'.format(coreutils.mem(data['validation']['x']) + coreutils.mem(data['validation']['y'])), flush=True)

    # Create TF session and graph
    graph = tf.Graph()
    sess = tf.Session(graph=graph)    
    
    for params in parameter_list:
        
        # Reset graph
        with graph.as_default():
            tf.reset_default_graph()
        
        # Create a DCN according to the spec
        dcn = getattr(compression, args.dcn)(sess, graph, training_spec['patch_size'], **params) 
        train_dcn({'dcn': dcn}, training_spec, data, args.out_dir)

if __name__ == "__main__":
    main()
