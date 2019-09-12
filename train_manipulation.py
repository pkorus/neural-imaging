#!/usr/bin/env python3
# coding: utf-8

# Basic imports
import os
import argparse

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper functions
from helpers import coreutils, dataset
from compression import afi


@coreutils.logCall
def batch_training(nip_model, camera_names=None, root_directory=None, loss_metric='L2', trainables=None,
                   jpeg_quality=None, jpeg_mode='soft', manipulations=None, dcn_model=None, downsampling='pool',
                   end_repetition=10, start_repetition=0, n_epochs=1001, patch=128,
                   use_pretrained=True, lambdas_nip=None, lambdas_dcn=None, nip_directory=None, split='120:30:4'):
    """
    Repeat training for multiple NIP regularization strengths.
    """

    if nip_model is None:
        raise FileNotFoundError('NIP model not specified!')

    if nip_directory is None or not os.path.isdir(nip_directory):
        raise FileNotFoundError('Invalid NIP snapshots directory: {}'.format(nip_directory))

    if root_directory is None:
        raise FileNotFoundError('Invalid root directory: {}'.format(root_directory))

    if not os.path.isdir(root_directory):
        os.makedirs(root_directory)

    # Lazy loading to minimize delays when checking cli parameters
    from training.manipulation import construct_models, train_manipulation_nip

    camera_names = camera_names or ['Nikon D90', 'Nikon D7000', 'Canon EOS 5D', 'Canon EOS 40D']

    training = {
        'use_pretrained_nip': use_pretrained,
        'n_epochs': n_epochs,
        'patch_size': patch,
        'batch_size': 20,
        'sampling_rate': 50,
        'learning_rate': 1e-4,
        'n_images': int(split.split(':')[0]),
        'v_images': int(split.split(':')[1]),
        'val_n_patches': int(split.split(':')[2]),
    }

    # Setup trainable elements and regularization ----------------------------------------------------------------------

    trainables = trainables if trainables is not None else set()
    for tr in trainables:
        if tr not in {'nip', 'dcn'}:
            raise ValueError('Invalid specifier of trainable elements: only nip, dcn allowed!')
            
    training['trainable'] = trainables

    if lambdas_nip is None or len(lambdas_nip) == 0:
        lambdas_nip = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1] if 'nip' in trainables else [0]

    else:
        lambdas_nip = [float(x) for x in lambdas_nip]

    if lambdas_dcn is None or len(lambdas_dcn) == 0:
        lambdas_dcn = [0.1, 0.05, 0.01, 0.005, 0.001] if 'dcn' in trainables else [0]
    else:
        lambdas_dcn = [float(x) for x in lambdas_dcn]

    if downsampling not in ['pool', 'bilinear', 'none']:
        raise ValueError('Unsupported channel down-sampling')

    if dcn_model is None and jpeg_quality is None:
        jpeg_quality = 50
        
    # Define the distribution channel ----------------------------------------------------------------------------------
    compression_params = {}
    if jpeg_quality is not None:
        compression_params['quality'] = jpeg_quality
        compression_params['rounding_approximation'] = jpeg_mode
    else:
        if dcn_model in afi.dcn_presets:
            dcn_model = afi.dcn_presets[dcn_model]
        compression_params['dirname'] = dcn_model

    if jpeg_quality is not None:
        compression = 'jpeg'
    elif dcn_model is not None:
        compression = 'dcn'
    else:
        compression = 'none'

    # Parse manipulations
    manipulations = manipulations or ['sharpen', 'resample', 'gaussian', 'jpeg']

    distribution_spec = {
        'downsampling': downsampling,
        'compression': compression,
        'compression_params': compression_params
    }

    # Construct the TF model
    tf_ops, distribution = construct_models(nip_model, patch_size=training['patch_size'], trainable=trainables, distribution=distribution_spec, manipulations=manipulations, loss_metric=loss_metric)

    for camera_name in camera_names:
        
        print('\n# Loading data for {}'.format(camera_name))
        
        training['camera_name'] = camera_name
        
        # Load the dataset
        if nip_model == 'ONet':
            # TODO Dirty hack - if the NIP model is the dummy empty model, load RGB images only
            data_directory = os.path.join('./data/rgb/', camera_name)
            patch_mul = 2
            load = 'y'
        else:
            # Otherwise, load (RAW, RGB) pairs for a specific camera
            data_directory = os.path.join('./data/raw/nip_training_data/', camera_name)
            patch_mul = 2
            load = 'xy'

        # Find available images
        data = dataset.IPDataset(data_directory, n_images=training['n_images'], v_images=training['v_images'], load=load, val_rgb_patch_size=patch_mul * training['patch_size'], val_n_patches=training['val_n_patches'])

        # data = dataset.IPDataset(data_directory, n_images=training['n_images'], v_images=training['v_images'], load=load, val_rgb_patch_size=training['patch_size'], val_n_patches=training['val_n_patches'])

        # Repeat evaluation
        for rep in range(start_repetition, end_repetition):
            for lr in lambdas_nip:
                for lc in lambdas_dcn:
                    training['lambda_nip'] = lr
                    training['lambda_dcn'] = lc
                    training['run_number'] = rep
                    train_manipulation_nip(tf_ops, training, distribution, data, {'root': root_directory, 'nip_snapshots': nip_directory})

                
def main():
    parser = argparse.ArgumentParser(description='NIP & FAN optimization for manipulation detection')

    group = parser.add_argument_group('general parameters')
    group.add_argument('--nip', dest='nip_model', action='store', required=True,
                        help='the NIP model (INet, UNet, DNet)')
    group.add_argument('--cam', dest='cameras', action='append',
                        help='add cameras for evaluation (repeat if needed)')
    group.add_argument('--manip', dest='manipulations', action='store', default='sharpen,resample,gaussian,jpeg',
                       help='Included manipulations, e.g., : {}'.format('sharpen,jpeg,resample,gaussian'))

    # Directories
    group = parser.add_argument_group('directories')
    group.add_argument('--dir', dest='root_dir', action='store', default='./data/raw/m_experimental/',
                        help='the root directory for storing results')
    group.add_argument('--nip-dir', dest='nip_directory', action='store', default='./data/raw/nip_model_snapshots/',
                        help='the root directory for storing results')

    # Training parameters
    group = parser.add_argument_group('training parameters')
    group.add_argument('--loss', dest='loss_metric', action='store', default='L2',
                        help='loss metric for the NIP (L2, L1, SSIM)')
    group.add_argument('--split', dest='split', action='store', default='120:30:4',
                        help='data split (#training:#validation:#validation_patches): e.g., 120:30:4')
    group.add_argument('--ln', dest='lambdas_nip', action='append',
                        help='set custom regularization strength for the NIP (repeat for multiple values)')
    group.add_argument('--lc', dest='lambdas_dcn', action='append',
                        help='set custom regularization strength for the DCN (repeat for multiple values)')
    group.add_argument('--train', dest='trainables', action='append',
                        help='add trainable elements (nip, dcn)')
    group.add_argument('--patch', dest='patch', action='store', default=256, type=int,
                        help='RGB patch size for NIP output (default 256)')

    # Training scope and progress
    group = parser.add_argument_group('training scope')
    group.add_argument('--scratch', dest='from_scratch', action='store_true', default=False,
                        help='train NIP from scratch (ignore pre-trained model)')
    group.add_argument('--start', dest='start', action='store', default=0, type=int,
                        help='first iteration (default 0)')
    group.add_argument('--end', dest='end', action='store', default=10, type=int,
                        help='last iteration (exclusive, default 10)')
    group.add_argument('--epochs', dest='epochs', action='store', default=1001, type=int,
                        help='number of epochs (default 1001)')

    # Distribution channel
    group = parser.add_argument_group('distribution channel')
    group.add_argument('--jpeg', dest='jpeg_quality', action='store', default=None, type=int,
                        help='JPEG quality level (distribution channel)')
    group.add_argument('--jpeg_mode', dest='jpeg_mode', action='store', default='soft',
                        help='JPEG approximation mode: sin, soft, harmonic')
    group.add_argument('--dcn', dest='dcn_model', action='store', default=None,
                        help='DCN compression model path')
    group.add_argument('--ds', dest='downsampling', action='store', default='pool',
                        help='Distribution channel sub-sampling: pool/bilinear/none')

    args = parser.parse_args()

    # Split manipulations
    args.manipulations = args.manipulations.strip().split(',')

    batch_training(args.nip_model, args.cameras, args.root_dir, args.loss_metric, args.trainables,
                   args.jpeg_quality, args.jpeg_mode, args.manipulations, args.dcn_model, args.downsampling, patch=args.patch // 2,
                   use_pretrained=not args.from_scratch, start_repetition=args.start, end_repetition=args.end, n_epochs=args.epochs,
                   nip_directory=args.nip_directory, split=args.split, lambdas_nip=args.lambdas_nip, lambdas_dcn=args.lambdas_dcn)


if __name__ == "__main__":
    main()
