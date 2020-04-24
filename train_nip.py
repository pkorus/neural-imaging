#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json
import argparse

import pandas as pd
import numpy as np

import helpers.debugging
from helpers import fsutil, dataset, utils
from training.pipeline import train_nip_model

# Set progress bar width
TQDM_WIDTH = 120

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_parameters(csv_file, metrics=('ssim', 'psnr', 'loss', 'params')):

    parameters = pd.DataFrame(columns=['scenario', 'label', 'active', 'run_group'])

    if csv_file is not None:
        parameters = parameters.append(pd.read_csv(csv_file), ignore_index=True, sort=True)

    if len(parameters) == 0:
        cli_params = {
            'scenario': np.nan,
            'label': 'command-line',
            'active': True,
            'run_group': np.nan
        }
        parameters = parameters.append(cli_params, ignore_index=True)

    # If requested, add columns to include validation results
    for key in metrics:
        parameters[key] = np.nan

    for col in parameters.columns:
        if col.startswith('@'):
            parameters[col] = parameters[col].apply(eval)
            parameters = parameters.rename(columns={col: col[1:]})

    return parameters

def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--nip', dest='nips', action='append', help='add NIP for training (repeat if needed)')
    parser.add_argument('--out', dest='out_dir', action='store', default='./data/models/nip',
                        help='output directory for storing trained NIP models')
    parser.add_argument('--data', dest='data_dir', action='store', default='./data/raw/training_data/',
                        help='input directory with training data (.npy and .png pairs)')
    parser.add_argument('--patch', dest='patch_size', action='store', default=128, type=int,
                        help='training patch size (RGB)')
    parser.add_argument('--epochs', dest='epochs', action='store', default=25000, type=int,
                        help='maximum number of training epochs')
    parser.add_argument('--ha', dest='hyperparams_args', default=None, help='Set hyper-parameters / override CSV settings if needed (JSON string)')
    parser.add_argument('--hp', dest='hyperparams_csv', default=None, help='CSV file with hyper-parameter configurations')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Resume training from last checkpoint, if possible')
    parser.add_argument('--split', dest='split', action='store', default='120:30:1',
                        help='data split with #training:#validation:#validation_patches - e.g., 120:30:1')
    parser.add_argument('--dry', dest='dry', action='store_true', default=False,
                        help='Dry run (no training - only does model setup)')
    parser.add_argument('--group', dest='run_group', action='store', type=int, default=None,
                        help='Specify run group (sub-selects scenarios for running)')
    parser.add_argument('--fill', dest='fill', action='store', default=None,
                        help='Path of the extended scenarios table with appended result columns')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    if not args.nips:
        print('At least one NIP needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    data_directory = os.path.join(args.data_dir, args.camera)
    out_directory_root = args.out_dir

    # List of hyper-parameters
    parameters = get_parameters(args.hyperparams_csv)

    if args.run_group is not None:
        parameters = parameters[parameters['run_group'] == args.run_group]

    # Select only active hyper-parameter configurations
    if len(parameters):
        parameters = parameters[parameters['active']].drop(columns=['active', 'run_group'])
    
    try:
        if args.hyperparams_args is not None:
            args.hyperparams_args = json.loads(args.hyperparams_args.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.hyperparams_args.replace('\'', '"'))
        sys.exit(2)

    if args.epochs < 0:
        convergence_threshold = 1e-6
        args.epochs = abs(args.epochs)
    else:
        convergence_threshold = None

    print('# Camera ISP Training')
    print('Camera          : {}'.format(args.camera))
    print('NIPs            : {}'.format(args.nips))
    print('Params (CSV)    : {}'.format(args.hyperparams_csv))
    print('Params override : {}'.format(args.hyperparams_args))
    print('Input           : {}'.format(data_directory))
    print('Output          : {}'.format(out_directory_root))
    print('Resume          : {}'.format(args.resume))
    print('Epochs          : {} {}'.format(args.epochs, '(convergence threshold {:.8f})'.format(convergence_threshold) if convergence_threshold is not None else '(fixed)'))

    print('\n# Hyper-parameter configurations [{} active configs]:\n'.format(len(parameters)))
    print(parameters)

    # Load training and validation data
    training_spec = {
        'seed': 1234,
        'n_images': int(args.split.split(':')[0]),
        'v_images': int(args.split.split(':')[1]),
        'valid_patches': int(args.split.split(':')[2]),
        'valid_patch_size': 256,
    }

    np.random.seed(training_spec['seed'])

    # Load and summarize the training data
    if not args.dry:
        print('\n# Dataset')
        data = dataset.Dataset(data_directory, n_images=training_spec['n_images'], v_images=training_spec['v_images'], load='xy', val_rgb_patch_size=training_spec['valid_patch_size'], val_n_patches=training_spec['valid_patches'])

        print(data.summary())

        for key in ['Training', 'Validation']:
            print('{:>16s} [{:5.1f} GB] : X -> {}, Y -> {} '.format(
                '{} data'.format(key),
                helpers.debugging.mem(data[key.lower()]['x']) + helpers.debugging.mem(data[key.lower()]['y']),
                data[key.lower()]['x'].shape,
                data[key.lower()]['y'].shape
            ), flush=True)

    # Lazy loading to prevent delays in basic CLI interaction
    from models import pipelines
    import tensorflow as tf

    # Train the Desired NIP Models
    model_log = {}
    print('\n# Training\n')
    for pipe in args.nips:

        for counter, (index, params) in enumerate(parameters.drop(columns=['scenario', 'label']).iterrows()):

            print('## {} : Scenario #{} - {} / {}'.format(pipe, index, counter + 1, len(parameters)))

            # Set hyper-parameters from the list
            params = {k: v for k, v in params.to_dict().items() if not utils.is_nan(v)}

            # Override hyper-parameters if requested
            if args.hyperparams_args is not None:
                print('info: overriding hyperparameters from the CLI-supplied JSON')
                params.update(args.hyperparams_args)

            if not issubclass(getattr(pipelines, pipe), pipelines.NIPModel):
                supported_nips = [x for x in dir(pipelines) if
                                x != 'NIPModel' and type(getattr(pipelines, x)) is type and issubclass(
                                    getattr(pipelines, x), pipelines.NIPModel)]
                raise ValueError('Invalid NIP model ({})! Available NIPs: ({})'.format(pipe, supported_nips))

            model = getattr(pipelines, pipe)(**params)

            if isinstance(model, pipelines.ClassicISP):
                with open('config/cameras.json') as f:
                    cameras = json.load(f)
                print('Configuring ISP to {}: {}'.format(args.camera, cameras[args.camera]))
                model.set_cfa_pattern(cameras[args.camera]['cfa'])
                model.set_srgb_conversion(np.array(cameras[args.camera]['srgb']))

            # Remember trained models
            model_code = model.model_code
            parameters.loc[index, 'model_code'] = model.model_code

            if model_code in model_log:
                print('WARNING - model {} already registered by scenario {}'.format(model_code, index))
                model_log[model_code].append(index)
            else:
                model_log[model_code] = [index]

            # Log the number of parameters, process a sample batch first to make sure the model is initialized
            # (does not happen when using custom tf.keras.Model classes)
            model.process(np.random.uniform(size=(1, 128, 128, 4)).astype(np.float32))
            parameters.loc[index, 'params'] = model.count_parameters()

            # Run training
            if not args.dry:
                out_dir = train_nip_model(model, args.camera, args.epochs, validation_loss_threshold=convergence_threshold, 
                    patch_size=args.patch_size, resume=args.resume, data=data, out_directory_root=args.out_dir)

            # Fill results
            if args.fill is not None:
                results_json = os.path.join(out_dir, 'progress.json')

                if os.path.isfile(results_json):

                    with open(results_json) as f:
                        results = json.load(f)

                    for key in ['ssim', 'psnr', 'loss']:
                        parameters.loc[index, key] = results['performance'][key]['validation'][-1]

    if args.fill is not None:

        if args.fill == '-':
            print('\n# Training Results')
            print(parameters.to_string())
        elif args.fill.endswith('.csv'):
            print('Saving the results to {}'.format(args.fill))
            parameters.to_csv(args.fill, index=False)
        else:
            raise ValueError('Invalid value for the output results file: {}'.format(args.fill))

    if args.dry:
        print('\n# List of instantiated models [{}]:'.format(len(model_log)))
        for index, key in enumerate(sorted(model_log.keys())):
            print('{}  {:3d}. {} -> {}'.format(' ' if len(model_log[key]) == 1 else '!', index, key, model_log[key]))


if __name__ == "__main__":
    main()
