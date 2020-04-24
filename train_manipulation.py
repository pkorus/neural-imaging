#!/usr/bin/env python3
# coding: utf-8

# Basic imports
import os
import re
import sys
import json
import argparse

from loguru import logger

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper functions
from helpers import fsutil, dataset, utils
from compression import codec


@utils.logCall
def batch_training(nip_model, camera_names=None, root_directory=None, loss_metric='L2', trainables=None,
                   jpeg_quality=None, jpeg_mode='soft', manipulations=None, dcn_model=None, downsampling='pool',
                   end_repetition=10, start_repetition=0, n_epochs=1001, patch=128, fan_args={},
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

    if jpeg_quality is not None:
        if re.match('^[0-9]+$', jpeg_quality):
            jpeg_quality = int(jpeg_quality)
        elif re.match('^[0-9\\,]+$', jpeg_quality):
            jpeg_quality = tuple(int(x) for x in re.findall('([0-9]+)', jpeg_quality))
        else:
            raise FileNotFoundError('Invalid JPEG quality: expecting a number or comma separated numbers & got: {}'.format(jpeg_quality))

    # Lazy loading to minimize delays when checking cli parameters
    from training.manipulation import train_manipulation_nip
    from workflows import manipulation_classification

    camera_names = camera_names or ['D90', 'D7000', 'EOS-5D', 'EOS-40D']

    training = {
        'use_pretrained_nip': use_pretrained,
        'n_epochs': n_epochs,
        'patch_size': patch,
        'batch_size': 20,
        'validation_schedule': 50,
        'learning_rate': 1e-4,
        'n_images': int(split.split(':')[0]),
        'v_images': int(split.split(':')[1]),
        'val_n_patches': int(split.split(':')[2]),
    }

    # Setup trainable elements and regularization -------------------------------------------------

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
        
    # Setup the distribution channel --------------------------------------------------------------
    if downsampling not in ['pool', 'bilinear', 'none']:
        raise ValueError('Unsupported channel down-sampling')

    if dcn_model is None and jpeg_quality is None:
        jpeg_quality = 50

    compression_params = {}
    if jpeg_quality is not None:
        compression_params['quality'] = jpeg_quality
        compression_params['codec'] = jpeg_mode
    else:
        compression_params['dirname'] = dcn_model

    if jpeg_quality is not None:
        compression = 'jpeg'
    elif dcn_model is not None:
        compression = 'dcn'
    else:
        compression = 'none'

    distribution = {
        'downsampling': downsampling,
        'compression': compression,
        'compression_params': compression_params
    }

    # Construct the workflow ----------------------------------------------------------------------
    manipulations = manipulations or ['sharpen', 'resample', 'gaussian', 'jpeg']

    flow = manipulation_classification.ManipulationClassification(nip_model, manipulations, distribution, fan_args, trainables, raw_patch_size=training['patch_size'])
    logger.info(f'Workflow: {flow.summary()}')
    logger.info(f'\n{flow.details()}')

    # Iterate over cameras and train the entire workflow ------------------------------------------ 
    for camera_name in camera_names:
        
        logger.info(f'Loading data for {camera_name}')
        training['camera_name'] = camera_name
        
        # Find the right dataset to load
        if nip_model == 'ONet':
            # TODO Dirty hack - if the NIP model is the dummy empty model, load RGB images only
            data_directory = os.path.join(root_directory, 'rgb', camera_name)
            patch_mul = 2
            load = 'y'
        else:
            # Otherwise, load (RAW, RGB) pairs for a specific camera
            data_directory = os.path.join(root_directory, 'raw', 'training_data', camera_name)
            patch_mul = 2
            load = 'xy'

        # If the target root directory has no training images, fallback to use the default root
        if not os.path.isdir(data_directory):
            logger.warning('Training images not found in the target root directory - using default root as image source')
            data_directory = data_directory.replace(root_directory, 'data/').replace('//', '/')

        # Load the image dataset
        data = dataset.Dataset(data_directory, n_images=training['n_images'], v_images=training['v_images'], load=load, val_rgb_patch_size=patch_mul * training['patch_size'], val_n_patches=training['val_n_patches'])

        logger.info('Training loop: {} repetitions / {} NIP lambdas {} / {} DCN lambdas {}'.format(
            end_repetition - start_repetition, len(lambdas_nip), lambdas_nip, len(lambdas_dcn), lambdas_dcn))
        
        # Repeat training with different loss weights
        for rep in range(start_repetition, end_repetition):
            for lr in lambdas_nip:
                for lc in lambdas_dcn:
                    training['lambda_nip'] = lr
                    training['lambda_dcn'] = lc
                    training['run_number'] = rep
                    train_manipulation_nip(flow, training, data, {'root': root_directory, 'nip_snapshots': nip_directory})

                
def main():
    parser = argparse.ArgumentParser(description='NIP & FAN optimization for manipulation detection')

    group = parser.add_argument_group('general parameters')
    group.add_argument('--nip', dest='nip_model', action='store', required=True,
                        help='the NIP model (INet, UNet, DNet)')
    group.add_argument('--cam', dest='cameras', action='append',
                        help='add cameras for evaluation (repeat if needed)')
    group.add_argument('--manip', dest='manipulations', action='store', default='sharpen,resample,gaussian,jpeg',
                       help='comma-sep. list of manipulations (:strength), e.g., : {}'.format('sharpen:1,jpeg:80,resample,gaussian'))
    group.add_argument('--fan', dest='fan_args', default=None,
                        help='Set hyper-parameters for the FAN model (JSON string)')

    # Directories
    group = parser.add_argument_group('directories')
    group.add_argument('--dir', dest='root_dir', action='store', default='./data/m/playground/',
                        help='the root directory for storing results')
    group.add_argument('--nip-dir', dest='nip_directory', action='store', default='./data/models/nip/',
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
    group.add_argument('--jpeg', dest='jpeg_quality', action='store', default=None, type=str,
                        help='JPEG quality level (distribution channel)')
    group.add_argument('--jpeg_mode', dest='jpeg_mode', action='store', default='soft',
                        help='JPEG approximation mode: sin, soft, harmonic')
    group.add_argument('--dcn', dest='dcn_model', action='store', default=None,
                        help='DCN compression model path')
    group.add_argument('--ds', dest='downsampling', action='store', default='pool',
                        help='Distribution channel sub-sampling: pool/bilinear/none')

    args = parser.parse_args()

    # Parse FAN args
    try:
        args.fan_args = json.loads(args.fan_args.replace('\'', '"')) if args.fan_args is not None else {}
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.hyperparams_args.replace('\'', '"'))
        sys.exit(2)

    # Split manipulations
    args.manipulations = args.manipulations.strip().split(',')

    batch_training(args.nip_model, args.cameras, args.root_dir, 
        args.loss_metric, args.trainables, args.jpeg_quality, args.jpeg_mode, 
        args.manipulations, args.dcn_model, args.downsampling, patch=args.patch // 2, fan_args=args.fan_args,
        use_pretrained=not args.from_scratch, start_repetition=args.start, end_repetition=args.end, n_epochs=args.epochs,
        nip_directory=args.nip_directory, split=args.split, lambdas_nip=args.lambdas_nip, lambdas_dcn=args.lambdas_dcn)


if __name__ == "__main__":
    main()
