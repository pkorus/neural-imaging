#!/usr/bin/env python3
# coding: utf-8
import os
import json
from collections import deque, OrderedDict

import numpy as np

from matplotlib.figure import Figure
from tqdm import tqdm
from helpers import metrics


# Set progress bar width
TQDM_WIDTH = 200

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def validate(model, data, out_directory, savefig=False, epoch=0, show_ref=False, loss_metric='L2'):
    
    ssims, psnrs, losss = [], [], []

    if loss_metric not in ['L2', 'L1', 'SSIM', 'MS-SSIM']:
        raise ValueError('Unsupported loss ({})!'.format(loss_metric))

    if savefig:
        images_x = np.minimum(data.count_validation, 20 if not show_ref else 10)
        images_y = np.ceil(data.count_validation / images_x)
        fig = Figure(figsize=(40, 1.1 * 40 / images_x * images_y * (1 if not show_ref else 0.5)))
        
    developed_out = np.zeros_like(data['validation']['y'], dtype=np.float32)

    for b in range(data.count_validation):

        # Fetch the next example and develop the RGB image
        example_x, example_y = data.next_validation_batch(b, 1)
        developed = model.process(example_x)
        developed = np.clip(developed, 0, 1)
        developed_out[b, :, :, :] = developed
        developed = developed[:, :, :, :].squeeze()        
        reference = example_y.squeeze()

        # Compute loss & quality metrics
        ssim = float(metrics.ssim(reference, developed))
        psnr = float(metrics.psnr(reference, developed))

        if loss_metric == 'L2':
            loss = metrics.mse(255 * reference, 255 * developed)
        elif loss_metric == 'L1':
            loss = metrics.mae(255 * reference, 255 * developed)
        elif loss_metric == 'SSIM':
            loss = 255 * (1 - metrics.ssim(reference, developed))
        else:
            raise ValueError('Unsupported loss ({})!'.format(loss_metric))

        ssims.append(ssim)
        psnrs.append(psnr)
        losss.append(loss)

        if savefig:
            ax = fig.add_subplot(images_y, images_x, b+1)
            if show_ref:
                ax.imshow(np.concatenate((reference, developed), axis=1))
            else:
                ax.imshow(developed)
            ax.set_xticks([])
            ax.set_yticks([])
            label_index = int(b // (data.count_validation / len(data.files['validation'])))
            ax.set_title('{} : {:.1f} dB / {:.2f}'.format(data.files['validation'][label_index], psnr, ssim), fontsize=6)

    if savefig:
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        fig.savefig(os.path.join(out_directory, 'validation_{:05d}.jpg'.format(epoch)), bbox_inches='tight', dpi=150)
        del fig
    
    return ssims, psnrs, losss, developed_out


# Show the training progress
def show_progress(isp, out_directory):
    from helpers import plots
    fig = plots.perf(isp.performance, ['training', 'validation'], figwidth=5)    
    fig.suptitle(isp.model_code)
    fig.savefig(os.path.join(out_directory, 'progress.png'), bbox_inches='tight', dpi=150)
    del fig


def save_progress(model, training_summary, out_directory):

    filename = os.path.join(out_directory, 'progress.json')
    output_stats = {
        'performance': model.performance, 
        'args': model.get_hyperparameters(), 
        'model': model.class_name, 
        'init': repr(model), 
        'summary': training_summary
    }
    with open(filename, 'w') as f:
        json.dump(output_stats, f, indent=4)


def train_nip_model(model, camera_name, n_epochs=10000, lr_schedule=None, validation_loss_threshold=1e-3,
                    validation_schedule=100, resume=False, patch_size=64, batch_size=20, data=None,
                    out_directory_root='./data/models/nip', save_best=False, discard='flat'):
    
    if data is None:
        raise ValueError('Training data seems not to be loaded!')
        
    try:
        batch_x, batch_y = data.next_training_batch(0, 5, patch_size * 2)
        if batch_x.shape != (5, patch_size, patch_size, 4) or batch_y.shape != (5, 2 * patch_size, 2 * patch_size, 3):
            raise ValueError('The training batch returned by the dataset instance is of invalid size!')

    except Exception as e:
        raise ValueError('Data set error: {}'.format(e))

    # Set up training output
    out_directory = os.path.join(out_directory_root, camera_name, model.model_code, model.scoped_name)

    if os.path.exists(out_directory) and not resume:
        print('WARNING directory {} exists, skipping...'.format(out_directory))
        return out_directory
    
    n_batches = data.count_training // batch_size
    n_tail = 5

    if not resume:
        losses_buf = deque(maxlen=10)
        loss_local = deque(maxlen=n_batches)
        start_epoch = 0
    else:
        # Find training summary
        summary_file = os.path.join(out_directory, 'progress.json')

        if not os.path.isfile(summary_file):
            raise FileNotFoundError('Could not open file {}'.format(summary_file))

        print('Resuming training from: {}'.format(summary_file))
        model.load_model(out_directory)

        with open(summary_file) as f:
            summary_data = json.load(f)

        # Read performance stats to date
        model.performance = summary_data['performance']

        # Initialize counters
        start_epoch = summary_data['summary']['Epoch']
        losses_buf = deque(maxlen=10)
        loss_local = deque(maxlen=n_batches)
        losses_buf.extend(model.performance['loss']['validation'][-10:])

    if lr_schedule is None:
        lr_schedule = {0: 1e-3, 1000: 1e-4, 2000: 1e-5}
    elif isinstance(lr_schedule, float):
        lr_schedule = {0: lr_schedule}
                
    # Collect and print training summary
    training_summary = OrderedDict()
    training_summary['Camera'] = camera_name
    training_summary['Architecture'] = model.summary()
    training_summary['Max epochs'] = n_epochs
    training_summary['Learning rate'] = lr_schedule
    training_summary['Training data size'] = data['training']['x'].shape
    training_summary['Validation data size'] = data['validation']['x'].shape
    training_summary['# batches'] = n_batches
    training_summary['Patch size'] = patch_size
    training_summary['Batch size'] = batch_size
    training_summary['Validation schedule'] = validation_schedule
    training_summary['Start epoch'] = start_epoch
    training_summary['Saved checkpoint'] = None
    training_summary['Discarding policy'] = discard
    training_summary['Output directory'] = out_directory

    print('\n## Training summary')
    for k, v in training_summary.items():
        print('{:30s}: {}'.format(k, v))
    print('', flush=True)
    
    with tqdm(total=n_epochs, ncols=TQDM_WIDTH, desc='{} for {}'.format(model.model_code, camera_name)) as pbar:
        pbar.update(start_epoch)
        
        learning_rate = 1e-3

        for epoch in range(start_epoch, n_epochs):
            
            if epoch in lr_schedule:
                learning_rate = min([learning_rate, lr_schedule[epoch]])

            for batch_id in range(n_batches):
                batch_x, batch_y = data.next_training_batch(batch_id, batch_size, patch_size, discard=discard)
                loss = model.training_step(batch_x, batch_y, learning_rate)
                loss_local.append(loss)

            model.performance['loss']['training'].append(float(np.mean(loss_local)))
            losses_buf.append(model.performance['loss']['training'][-1])

            if epoch == start_epoch:
                developed = np.zeros_like(data['validation']['y'], dtype=np.float32)

            if epoch % validation_schedule == 0:
                # Use the current model to develop images in the validation set
                developed_old = developed
                ssims, psnrs, v_losses, developed = validate(model, data, out_directory, True, epoch, True, loss_metric=model.loss_metric)
                model.performance['ssim']['validation'].append(float(np.mean(ssims)))
                model.performance['psnr']['validation'].append(float(np.mean(psnrs)))
                model.performance['loss']['validation'].append(float(np.mean(v_losses)))

                # Compare the current images to the ones from a previous model iteration
                dmses = []
                for v in range(data['validation']['x'].shape[0]):
                    dmses.append(metrics.mse(developed_old[v, :, :, :], developed[v, :, :, :]))

                model.performance['dmse']['validation'].append(np.mean(dmses))

                # Generate progress summary
                training_summary['Epoch'] = epoch                
                # show_progress(model, out_directory)
                save_progress(model, training_summary, out_directory)
                
                if not save_best or (len(model.performance['loss']['validation']) > 2 and model.performance['loss']['validation'][-1] <= min(model.performance['loss']['validation'])):
                    training_summary['Saved checkpoint'] = epoch
                    model.save_model(out_directory, epoch, quiet=True)
                
                # If model deteriorated by more than 20%, drop the learning rate
                if len(model.performance['loss']['validation']) > 5:
                    if model.performance['loss']['validation'][-1] > 1.2 * min(model.performance['loss']['validation']):
                        learning_rate = learning_rate * 0.95
                        learning_rate = max((learning_rate, 1e-7))
                    
                # Check for convergence
                if validation_loss_threshold is not None and len(model.performance['loss']['validation']) > 10:
                    current = np.mean(model.performance['loss']['validation'][-n_tail:-1])
                    previous = np.mean(model.performance['loss']['validation'][-(n_tail + 1):-2])
                    vloss_change = abs((current - previous) / previous)

                    if vloss_change < validation_loss_threshold:
                        print('Early stopping - the model converged, validation loss change {}'.format(vloss_change))
                        break
                else:
                    vloss_change = np.nan

                progress_dict = {
                    'psnr': model.performance['psnr']['validation'][-1],
                    'ssim': model.performance['ssim']['validation'][-1],
                    'dmse': np.log10(model.performance['dmse']['validation'][-1]),
                }

            progress_dict['loss'] = np.mean(losses_buf)
            progress_dict['lr'] = learning_rate

            if not np.isnan(vloss_change):
                progress_dict['dloss'] = vloss_change

            pbar.set_postfix(**progress_dict)
            pbar.update(1)

    training_summary['Epoch'] = epoch
    if not save_best or (model.performance['loss']['validation'][-1] <= min(model.performance['loss']['validation'])):
        training_summary['Saved checkpoint'] = epoch
        model.save_model(out_directory, epoch)
    show_progress(model, out_directory)
    save_progress(model, training_summary, out_directory)

    return out_directory
