#!/usr/bin/env python
# coding: utf-8

# Basic imports
import os
import shutil
import numpy as np
import tqdm
from collections import deque, OrderedDict

from loguru import logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Disable unimportant logging and import TF
from helpers import debugging, utils
from models import compression, jpeg
from training import validation


def default_training_specs():
    return {
        'use_pretrained_nip': True,
        'patch_size': 64,
        'batch_size': 10,
        'validation_schedule': 50,
        'n_epochs': 1001,
        'learning_rate': 1e-4,
        'run_number': 0,
        'lambda_nip': 0.1,
        'lambda_dcn': 0,
        'augment': False,
    }


def train_manipulation_nip(flow, training, data, directories=None, overwrite=False):
    """
    Training procedure for the manipulation classification workflow. This function is mainly the boilerplate
    code that handles parameter sanitization, validation, saving and showing progress, etc. A minimalistic training
    code would be just (for a single epoch):

    for batch_id in range(n_batches):
        batch_x, batch_y = data.next_training_batch(batch_id, batch_size, rgb_patch_size)
        loss, losses = flow.training_step(batch_x, batch_y, lambda_nip, lambda_dcn, learning_rate)
    
    See: workflows.manipulation_classification
    
    :param flow: an instance of the ManipulationClassification workflow
    :param training: dict with training setup (see function default_training_specs() for an example)
    {
        camera_name          : name of the camera, used to re-load weights of the NIP for a given camera
        use_pretrained_nip   : boolean flag to enable/disable loading a pre-trained model
        lambda_nip           : regularization strength to control the trade-off between objectives (image quality)
        lambda_dcn           : regularization strength to control the trade-off between objectives (compression quality)
        run_number           : number of the current run ()
        n_epochs             : number of training epochs
        learning_rate        : value of the learning rate
    }
    :param data: instance of the Dataset
    :param data: instance of the Dataset
    :param directories: dict with directories for training output & NIP models 
    {
        root             : root output directory for saving training progress and model snapshots (default: './data/m/')
        nip_snapshots    : root directory with pre-trained NIP models (default: './data/models/nip/')
    }
    """
    
    # Apply default settings
    directories_def = {'root': './data/m/', 'nip_snapshots': './data/models/nip/'}
    if directories is not None:
        directories_def.update(directories)
    directories = directories_def

    training_defaults = default_training_specs()
    if training is not None:
        training_defaults.update(training)
    training = training_defaults

    # Check if all needed options are set in training specification
    required_keys = {'camera_name', 'use_pretrained_nip', 'lambda_nip', 'lambda_dcn', 'run_number', 'n_epochs', 'learning_rate', 'augment'}

    # logger.debug(f'Training spec {training.keys()}')

    if any([x not in training for x in required_keys]):
        raise RuntimeError('Missing keys in the training dictionary! {}'.format(required_keys.difference(training.keys())))

    if data is None:
        raise ValueError('Training data seems not to be loaded!')

    try:
        if data.is_raw_and_rgb():
            batch_x, batch_y = data.next_training_batch(0, 1, training['patch_size'] * 2)
            if batch_x.shape != (1, training['patch_size'], training['patch_size'], 4) or batch_y.shape != (1, 2 * training['patch_size'], 2 * training['patch_size'], 3):
                raise ValueError(f'The RAW+RGB training batch is of invalid size! {batch_x.shape}')
        else:
            batch_x = data.next_training_batch(0, 1, training['patch_size'] * 2)
            if batch_x.shape != (1, 2 * training['patch_size'], 2 * training['patch_size'], 3):
                raise ValueError(f'The RGB training batch is of invalid size! {batch_x.shape}')

    except Exception as e:
        raise ValueError('Data set error: {}'.format(e))

    logger.info('Training manipulation classification: cam={} / lr={:.4f} / run={:3d} / epochs={}, root={}'.format(
        training['camera_name'], training['lambda_nip'], training['run_number'], training['n_epochs'],
        directories['root']), flush=True)

    # Construct output directory - some example paths:
    #  root / camera_name / *Net / fixed-nip / fixed-codec / 001 /
    #  root / camera_name / *Net / lr-0.1000 / lc-0.1000 / 001 /
    #  root / camera_name / *Net / fixed-nip / lc-0.1000 / 001 /
    nip_save_dir = [directories['root'], training['camera_name'], flow.nip.class_name]
    if flow.is_trainable('nip'):
        nip_save_dir.append('ln-{:0.4f}'.format(training['lambda_nip']))
    else:
        nip_save_dir.append('fixed-nip')
    if flow.is_trainable('dcn'):
        nip_save_dir.append('lc-{:0.4f}'.format(training['lambda_dcn']))
    else:
        nip_save_dir.append('fixed-codec')
    nip_save_dir.append('{:03d}'.format(training['run_number']))
    
    nip_save_dir = os.path.join(*nip_save_dir)
    logger.info(f'(progress) -> {nip_save_dir}')

    model_directory = os.path.join(nip_save_dir, 'models')
    logger.info(f'(model) ----> {model_directory}')

    if os.path.exists(nip_save_dir) and not overwrite:
        logger.debug('Directory exists, skipping...')
        return model_directory

    if flow.is_trainable('nip') and flow.nip.count_parameters() == 0:
        raise ValueError('It looks like you`re trying to optimize a NIP with no trainable parameters!')

    # Basic setup
    learning_rate_decay_schedule = 100
    learning_rate_decay_rate = 0.90
    learning_rate = training['learning_rate']
    n_batches = data.count_training // training['batch_size']

    # Load different NIP weights if specified
    if training['use_pretrained_nip'] and flow.nip.count_parameters() > 0:
        nip_dirname = os.path.join(directories['nip_snapshots'], training['camera_name'], flow.nip.model_code)
        logger.debug(f'Loading camera model from {nip_dirname}')
        flow.nip.load_model(nip_dirname)

    model_list = ['nip', 'fan']

    # Containers for storing loss progression
    loss_epoch = {key: deque(maxlen=n_batches) for key in model_list}
    loss_epoch['similarity-loss'] = deque(maxlen=n_batches)
    loss_last_k_epochs = {key: deque(maxlen=10) for key in model_list}
    loss_last_k_epochs['similarity-loss'] = deque(maxlen=n_batches)
    
    # Collect memory usage (seems to be leaking in matplotlib)
    collect_memory_stats = {'tf': False, 'ram': False}
    memory = {'tf-ram': [], 'tf-vars': [], 'cpu-proc': [], 'cpu-resource': [] }

    # Collect and print training summary
    training_summary = OrderedDict()
    training_summary['Problem'] = flow.summary()
    training_summary['Dataset'] = data.summary()
    training_summary['Camera name'] = training['camera_name']
    training_summary['Classes'] = f'{flow._forensics_classes}'
    training_summary['FAN model'] = flow.fan.summary()
    training_summary['NIP model'] = flow.nip.summary()
    training_summary['Channel Downsampling'] = flow._distribution['downsampling']
    training_summary['Channel Compression'] = flow.codec.summary() if flow.codec is not None else 'n/a'
    training_summary['Channel Compression Parameters'] = str(flow._distribution['compression_params'])
    training_summary['Joint optimization'] = f'{flow.trainable_models}'
    training_summary['NIP Regularization'] = utils.format_number(training['lambda_nip'])
    training_summary['DCN Regularization'] = utils.format_number(training['lambda_dcn'])
    training_summary['NIP loss'] = f'{flow.nip.loss_metric}'
    training_summary['Use pre-trained NIP'] = str(training['use_pretrained_nip'])
    training_summary['# Epochs'] = utils.format_number(training['n_epochs'])
    training_summary['Patch size'] = utils.format_number(training['patch_size'])
    training_summary['Batch size'] = utils.format_number(training['batch_size'])
    training_summary['Learning rate'] = utils.format_number(training['learning_rate'])
    training_summary['Learning rate decay schedule'] = utils.format_number(learning_rate_decay_schedule)
    training_summary['Learning rate decay rate'] = utils.format_number(learning_rate_decay_rate)
    training_summary['Validation schedule'] = training['validation_schedule']
    training_summary['Augmentation'] = str(training['augment'])
    training_summary['# train. images'] = utils.format_number(data.count_training)
    training_summary['# valid. images'] = utils.format_number(data.count_validation)
    training_summary['Batch shape'] = f'{batch_x.shape}'
    training_summary['NIP input patch'] = f'{flow.nip.x.shape}'
    training_summary['NIP output patch'] = f'{flow.nip.y.shape}'
    training_summary['FAN input patch'] = f'{flow.fan.x.shape}'
    if any(collect_memory_stats.values()):
        training_summary['memory_consumption'] = memory

    print('')
    for k, v in training_summary.items():
        print('{:30s}: {}'.format(k, v))
    print('', flush=True)

    with tqdm.tqdm(total=training['n_epochs'], ncols=120, desc='Train') as pbar:
        
        epoch = 0
        conf = np.identity(flow.n_classes)

        for epoch in range(0, training['n_epochs']):

            for batch_id in range(n_batches):

                # Extract random patches for the current batch of images
                if data._loaded_data == 'xy':
                    batch_x, batch_y = data.next_training_batch(batch_id, training['batch_size'], 2 * training['patch_size'])
                else:
                    batch_x = data.next_training_batch(batch_id, training['batch_size'], 2 * training['patch_size'])
                    batch_y = batch_x

                comb_loss, comp_loss = flow.training_step(batch_x, batch_y, training['lambda_nip'], training['lambda_dcn'], training['augment'], learning_rate)
                    
                loss_epoch['fan'].append(comb_loss)
                loss_epoch['nip'].append(comp_loss['nip'])

            # Average and record loss values
            for model_name, model in zip(model_list, [flow.nip, flow.fan]):
                model.log_metric('loss', 'training', loss_epoch[model_name])
                loss_last_k_epochs[model_name].append(model.pop_metric('loss', 'training'))

            # Model validation
            if epoch % training['validation_schedule'] == 0:

                # Validate the forensics analysis network
                accuracy, conf = validation.validate_fan(flow, data)
                flow.fan.log_metric('accuracy', 'validation', accuracy)
                flow.fan.performance['confusion'] = conf.tolist()

                # Validate the NIP model
                if flow.is_trainable('nip'):

                    values = validation.validate_nip(flow.nip, data, nip_save_dir, epoch=epoch, show_ref=True, loss_type=flow.nip.loss_metric)
                    
                    for metric, val_array in zip(['ssim', 'psnr', 'loss'], values):
                        flow.nip.log_metric(metric, 'validation', val_array)
                        
                # Validate the DCN model
                if flow.is_trainable('dcn'):
                    if isinstance(flow.codec, compression.DCN):
                        values = validation.validate_dcn(flow.codec, data, nip_save_dir, epoch=epoch, show_ref=True)
                    elif isinstance(flow.codec, jpeg.JPEG):
                        values = validation.validate_jpeg(flow.codec, data)
                    else:
                        raise NotImplementedError('Validation for {} codec doesn\'t seem to be implemented'.format(flow.codec))
                    
                    for metric, value in values.items():
                        flow.codec.log_metric(metric, 'validation', value)
                
                # Save progress stats
                validation.save_training_progress(training_summary, flow, nip_save_dir, quiet=True)

                # Save models
                flow.fan.save_model(os.path.join(model_directory, flow.fan.scoped_name), epoch, quiet=True)

                if flow.is_trainable('nip'):
                    flow.nip.save_model(os.path.join(model_directory, flow.nip.scoped_name), epoch, quiet=True)

                if isinstance(flow.codec, compression.DCN) and flow.is_trainable('dcn'):
                    flow.codec.save_model(os.path.join(model_directory, flow.codec.scoped_name), epoch, quiet=True)

                # Monitor memory usage - used to have memory leaks in matplotlib
                if collect_memory_stats['ram']:
                    memory['cpu-proc'].append(round(debugging.memory_usage_proc(), 1))
                    memory['cpu-resource'].append(round(debugging.memory_usage_resource(), 1))

            if epoch % learning_rate_decay_schedule == 0:
                learning_rate *= learning_rate_decay_rate

            # Update the progress bar
            progress_stats = {
                'fan': np.mean(loss_last_k_epochs['fan']),
                'acc': flow.fan.performance['accuracy']['validation'][-1],
            }
            
            if flow.is_trainable('dcn'):
                progress_stats['codec'] = flow.codec.performance['ssim']['validation'][-1]
                progress_stats['H'] = flow.codec.performance['entropy']['validation'][-1]

            if np.mean(loss_last_k_epochs['nip']) > 0:
                progress_stats['nip'] = np.mean(loss_last_k_epochs['nip']).round(2)
            
            if flow.is_trainable('dcn'):
                progress_stats['codec'] = flow.codec.performance['ssim']['validation'][-1]
                progress_stats['H'] = flow.codec.performance['entropy']['validation'][-1]

            if flow.is_trainable('dcn') and isinstance(flow.codec, jpeg.JPEG):
                progress_stats['JPEG'] = flow.codec.estimate_qf()

            if len(flow.nip.performance['psnr']['validation']) > 0:
                progress_stats['psnr'] = flow.nip.performance['psnr']['validation'][-1]

            if collect_memory_stats['ram']:
                progress_stats['ram'] = round(memory['cpu-proc'][-1]//1024, 2)

            pbar.set_postfix(**progress_stats)
            pbar.update(1)

    # Final validation and plotting
    accuracy, conf = validation.validate_fan(flow, data)
    flow.fan.performance['accuracy']['validation'].append(accuracy)
    flow.fan.performance['confusion'] = conf.tolist()

    if flow.is_trainable('nip'):
        values = validation.validate_nip(flow.nip, data, nip_save_dir, epoch=epoch, show_ref=True, loss_type='L2')
        for metric, val_array in zip(['ssim', 'psnr', 'loss'], values):
            flow.nip.log_metric(metric, 'validation', val_array)

    if flow.is_trainable('dcn'):
        if isinstance(flow.codec, compression.DCN):
            values = validation.validate_dcn(flow.codec, data, nip_save_dir, epoch=epoch, show_ref=True)
            for metric, val_array in values.items():
                flow.codec.log_metric(metric, 'validation', val_array)

    # Save model progress
    validation.save_training_progress(training_summary, flow, nip_save_dir)

    # Visualize current progress
    validation.visualize_manipulation_training(flow, epoch, nip_save_dir)

    # Save models
    logger.info('Saving models...')

    flow.fan.save_model(os.path.join(model_directory, flow.fan.scoped_name), epoch)

    if flow.is_trainable('nip'):
        flow.nip.save_model(os.path.join(model_directory, flow.nip.scoped_name), epoch)

    if flow.is_trainable('dcn') and isinstance(flow.codec, compression.DCN):
        flow.codec.save_model(os.path.join(model_directory, flow.codec.scoped_name), epoch)
        shutil.copyfile(os.path.join(flow._distribution['compression_params']['dirname'], flow.codec.scoped_name, 'progress.json'),
                        os.path.join(model_directory, flow.codec.scoped_name, 'progress.json'))

    return model_directory
