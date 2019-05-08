#!/usr/bin/env python
# coding: utf-8

# Basic imports
import os
import argparse

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper functions
from helpers import coreutils, dataset
from training.manipulation import construct_models, train_manipulation_nip


@coreutils.logCall
def batch_training(nip_model, camera_names=None, root_directory=None, loss_metric='L2', jpeg_quality=50, jpeg_mode='sin', use_pretrained=True, end_repetition=10, start_repetition=0, n_epochs=1001, nip_directory=None):
    """
    Repeat training for multiple NIP regularization strengths.
    """

    training = {
        'use_pretrained_nip': use_pretrained,
        'n_epochs': n_epochs,
        'patch_size': 128,
        'batch_size': 20,
        'sampling_rate': 50,
        'learning_rate': 1e-4,
        'n_images': 120,
        'v_images': 30,
        'val_n_patches': 4
    }

    if root_directory is None or not os.path.isdir(root_directory):
        raise FileNotFoundError('Invalid root directory: {}'.format(root_directory))

    if nip_directory is None or not os.path.isdir(nip_directory):
        raise FileNotFoundError('Invalid NIP snapshots directory: {}'.format(nip_directory))

    # Experiment setup
    camera_names = camera_names or ['Nikon D90', 'Nikon D7000', 'Canon EOS 5D', 'Canon EOS 40D']
    regularization_strengths = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1]

    # Construct the TF model
    tf_ops, distribution = construct_models(nip_model, distribution_jpeg=jpeg_quality, loss_metric=loss_metric, jpeg_approx=jpeg_mode)

    for camera_name in camera_names:
        
        print('\n# Loading data for camera {}'.format(camera_name))
        
        training['camera_name'] = camera_name
        
        # Load the dataset for the given camera
        data_directory = os.path.join('./data/raw/nip_training_data/', camera_name)

        # Find available images
        data = dataset.IPDataset(data_directory, n_images=training['n_images'], v_images=training['v_images'], load='xy', val_rgb_patch_size=2 * training['patch_size'], val_n_patches=training['val_n_patches'])

        # Repeat evaluation
        for rep in range(start_repetition, end_repetition):
            for reg in regularization_strengths:
                training['nip_weight'] = reg
                training['run_number'] = rep
                train_manipulation_nip(tf_ops, training, distribution, data, {'root': root_directory, 'nip_snapshots': nip_directory})

                
def main():
    parser = argparse.ArgumentParser(description='NIP & FAN optimization for manipulation detection')
    parser.add_argument('--nip', dest='nip_model', action='store',
                        help='the NIP model (INet, UNet, DNet)')
    parser.add_argument('--quality', dest='jpeg_quality', action='store', default=50, type=int,
                        help='JPEG quality level in the distribution channel')
    parser.add_argument('--jpeg', dest='jpeg_mode', action='store', default='sin',
                        help='JPEG approximation mode: sin, soft, harmonic')
    parser.add_argument('--dir', dest='root_dir', action='store', default='./data/raw/train_manipulation/',
                        help='the root directory for storing results')
    parser.add_argument('--nip-dir', dest='nip_directory', action='store', default='./data/raw/nip_model_snapshots/',
                        help='the root directory for storing results')
    parser.add_argument('--cam', dest='cameras', action='append',
                        help='add cameras for evaluation (repeat if needed)')
    parser.add_argument('--loss', dest='loss_metric', action='store', default='L2',
                        help='loss metric for the NIP (L2, L1, SSIM)')
    parser.add_argument('--scratch', dest='from_scratch', action='store_true', default=False,
                        help='train NIP from scratch (ignore pre-trained model)')
    parser.add_argument('--start', dest='start', action='store', default=0, type=int,
                        help='first iteration (default 0)')
    parser.add_argument('--end', dest='end', action='store', default=10, type=int,
                        help='last iteration (exclusive, default 10)')
    parser.add_argument('--epochs', dest='epochs', action='store', default=1001, type=int,
                        help='number of epochs (default 1001)')

    args = parser.parse_args()

    batch_training(args.nip_model, args.cameras, args.root_dir, args.loss_metric, args.jpeg_quality, args.jpeg_mode, not args.from_scratch, start_repetition=args.start, end_repetition=args.end, n_epochs=args.epochs, nip_directory=args.nip_directory)


if __name__ == "__main__":
    main()
