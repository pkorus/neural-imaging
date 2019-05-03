#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json
import argparse
from collections import deque, OrderedDict

import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from skimage.measure import compare_ssim, compare_psnr, compare_mse

from helpers import coreutils
from training import train_nip_model

# Set progress bar width
TQDM_WIDTH = 120

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():

    global data_x, data_y, valid_x, valid_y, camera_name, out_directory_root, val_files

    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--nip', dest='nips', action='append', help='add NIP for training (repeat if needed)')
    parser.add_argument('--out', dest='out_dir', action='store', default='./data/raw/nip_model_snapshots',
                        help='output directory for storing trained NIP models')
    parser.add_argument('--data', dest='data_dir', action='store', default='./data/raw/nip_training_data/',
                        help='input directory with training data (.npy and .png pairs)')
    parser.add_argument('--patch', dest='patch_size', action='store', default=64, type=int,
                        help='training patch size')
    parser.add_argument('--epochs', dest='epochs', action='store', default=25000, type=int,
                        help='maximum number of training epochs')
    parser.add_argument('--params', dest='nip_params', default=None, help='Extra parameters for NIP constructor (JSON string)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Resume training from last checkpoint, if possible')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    if not args.nips:
        print('At least one NIP needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    # Lazy loading after parameter sanitization finished
    from helpers import loading

    data_directory = os.path.join(args.data_dir, args.camera)
    out_directory_root = args.out_dir
    try:
        if args.nip_params is not None:
            args.nip_params = json.loads(args.nip_params.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.nip_params.replace('\'', '"'))
        sys.exit(2)

    print('Camera : {}'.format(args.camera))
    print('NIPs   : {}'.format(args.nips))
    print('Params : {}'.format(args.nip_params))
    print('Input  : {}'.format(data_directory))
    print('Output : {}'.format(out_directory_root))
    print('Resume : {}'.format(args.resume))

    # Load training and validation data
    training_spec = {
        'seed': 1234,
        'n_images': 120,
        'v_images': 30,
        'valid_patch_size': 256,
        'valid_patches': 1
    }

    np.random.seed(training_spec['seed'])
    
    data = dataset.IPDataset(data_directory, training_spec['n_images'], training_spec['v_images'], load='xy', val_patch_size=training['patch_size'])


    # Find available images
#     files, val_files = loading.discover_files(data_directory, training_spec['n_images'], training_spec['v_images'])

    # Load training / validation data
#     data = {
#         'train': loading.load_images(files, data_directory, load='xy'),
#         'valid': loading.load_patches(val_files, data_directory, valid_patch_size, n_patches, discard_flat=True, load='xy')
#     }
    
    camera_name = args.camera

    print('Training data shape (X)   : {}'.format(data['training']['x'].shape))
    print('Training data size        : {:.1f} GB'.format(coreutils.mem(data['training']['x']) + coreutils.mem(data['training']['y'])), flush=True)
    print('Validation data shape (X) : {}'.format(data['validation']['x'].shape))
    print('Validation data size      : {:.1f} GB'.format(coreutils.mem(data['validation']['x']) + coreutils.mem(data['validation']['y'])), flush=True)

    # Train the Desired NIP Models
    for pipe in args.nips:
        train_nip_model(pipe, args.epochs, 1e-4, patch_size=args.patch_size, resume=args.resume, nip_params=args.nip_params)


if __name__ == "__main__":
    main()
