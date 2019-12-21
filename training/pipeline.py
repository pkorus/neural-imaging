#!/usr/bin/env python3
# coding: utf-8
import os
import json
from collections import deque, OrderedDict

import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from skimage.measure import compare_ssim, compare_psnr, compare_mse

# Set progress bar width
TQDM_WIDTH = 120

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def validate(model, data, out_directory, savefig=False, epoch=0, show_ref=False, loss_metric='L2'):
    
    ssims, psnrs, losss = [], [], []

    if loss_metric not in ['L2', 'L1']:
        raise ValueError('Unsupported loss ({})!'.format(loss_metric))

    if savefig:
        images_x = np.minimum(data.count_validation, 10 if not show_ref else 5)
        images_y = np.ceil(data.count_validation / images_x)
        plt.figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))
        
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
        ssim = float(compare_ssim(reference, developed, multichannel=True))
        psnr = float(compare_psnr(reference, developed))

        if loss_metric == 'L2':
            loss = float(np.mean(np.power(255.0*reference - 255.0*developed, 2.0)))
        elif loss_metric == 'L1':
            loss = float(np.mean(np.abs(255.0*reference - 255.0*developed)))
        else:
            raise ValueError('Unsupported loss ({})!'.format(loss_metric))

        ssims.append(ssim)
        psnrs.append(psnr)
        losss.append(loss)

        if savefig:
            plt.subplot(images_y, images_x, b+1)
            if show_ref:
                plt.imshow(np.concatenate((reference, developed), axis=1))
            else:
                plt.imshow(developed)
            plt.xticks([])
            plt.yticks([])
            label_index = int(b // (data.count_validation / len(data.files['validation'])))
            plt.title('{} : {:.1f} dB / {:.2f}'.format(data.files['validation'][label_index], psnr, ssim), fontsize=6)

    if savefig:
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        plt.savefig(os.path.join(out_directory, 'validation_{:05d}.jpg'.format(epoch)), bbox_inches='tight', dpi=150)
        plt.close()
    
    return ssims, psnrs, losss, developed_out


# Show the training progress
def visualize_progress(arch, performance, patch_size, camera_name, out_directory, plot=False, sampling_rate=100):
    from helpers import utils

    v_range = np.arange(0, sampling_rate*len(performance['ssim']['validation']), sampling_rate)

    plt.figure(figsize=(16, 6))
    plt.subplot(2,2,1)
    plt.semilogy(performance['loss']['training'], alpha=0.15)
    plt.plot(utils.ma_conv(performance['loss']['training'], np.maximum(10, len(performance['loss']['training']) // 25)))
    plt.plot(v_range, np.array(performance['loss']['validation']), '.-', alpha=0.5)
    plt.ylabel('Loss')
    plt.legend(['loss (batch)', 'mov. avg loss (batch)', 'loss (valid.)'])

    if len(performance['loss']['validation']) > 10:
        n_tail = 5
        current = np.mean(performance['loss']['validation'][-n_tail:-1])
        previous = np.mean(performance['loss']['validation'][-(n_tail + 1):-2])
        vloss_change = abs((current - previous) / previous)
        plt.title('Validation loss change: {:.6f}'.format(vloss_change))
    
    plt.subplot(2,2,2)
    plt.plot(v_range, performance['ssim']['validation'], '.-', alpha=0.5)
    plt.ylabel('SSIM')
    
    plt.subplot(2,2,3)    
    plt.plot(v_range, np.array(performance['psnr']['validation']), '.-', alpha=0.5)
    plt.ylabel('PSNR')
    
    plt.subplot(2,2,4)
    plt.semilogy(v_range, np.array(performance['dmse']['validation']), '.-', alpha=0.5)
    plt.ylabel('$\Delta$ MSE from last')
    
    plt.suptitle('{} for {} ({}px): PSNR={:.1f}, SSIM={:.2f}'.format(arch, camera_name, patch_size, performance['psnr']['validation'][-1], performance['ssim']['validation'][-1]))
    if plot:
        plt.show()
    else:
        plt.savefig(os.path.join(out_directory, 'progress.png'), bbox_inches='tight', dpi=150)
        plt.close()


def save_progress(performance, training_summary, out_directory):

    filename = os.path.join(out_directory, 'progress.json')
    output_stats = {'performance': performance}
    output_stats.update(training_summary)
    with open(filename, 'w') as f:
        json.dump(output_stats, f, indent=4)


def train_nip_model(model, camera_name, n_epochs=10000, validation_loss_threshold=1e-3, sampling_rate=100, resume=False, patch_size=64, batch_size=20, data=None, out_directory_root='./data/models/nip'):
    
    if data is None:
        raise ValueError('Training data seems not to be loaded!')
        
    try:
        batch_x, batch_y = data.next_training_batch(0, 5, patch_size * 2)
        if batch_x.shape != (5, patch_size, patch_size, 4) or batch_y.shape != (5, 2 * patch_size, 2 * patch_size, 3):
            raise ValueError('The training batch returned by the dataset instance is of invalid size!')

    except Exception as e:
        raise ValueError('Data set error: {}'.format(e))

    # Set up training output
    out_directory = os.path.join(out_directory_root, camera_name, model.scoped_name)

    if os.path.exists(out_directory) and not resume:
        print('WARNING directory {} exists, skipping...'.format(out_directory))
        return

    # Limit the number of checkpoints to 5
    model.saver.saver_def.max_to_keep = 5
    model.saver._max_to_keep = 5
    
    n_batches = data.count_training // batch_size
    learning_rate = 1e-4
    n_tail = 5

    if not resume:
        losses_buf = deque(maxlen=10)
        loss_local = deque(maxlen=n_batches)
        model.init()
        start_epoch = 0
    else:
        # Find training summary
        summary_file = os.path.join(out_directory_root, camera_name, model.scoped_name, 'progress.json')

        if not os.path.isfile(summary_file):
            raise FileNotFoundError('Could not open file {}'.format(summary_file))

        print('Resuming training from: {}'.format(summary_file))
        model.load_model(os.path.join(out_directory_root, camera_name))

        with open(summary_file) as f:
            summary_data = json.load(f)

        # Read performance stats to date
        model.performance = summary_data['performance']

        # Initialize counters
        start_epoch = summary_data['Epoch']
        losses_buf = deque(maxlen=10)
        loss_local = deque(maxlen=n_batches)
        losses_buf.extend(model.performance['loss']['validation'][-10:])

    # Collect and print training summary
    training_summary = OrderedDict()
    training_summary['Camera'] = camera_name
    training_summary['Architecture'] = model.summary()
    training_summary['Max epochs'] = n_epochs
    training_summary['Learning rate'] = learning_rate
    training_summary['Training data size'] = data['training']['x'].shape
    training_summary['Validation data size'] = data['validation']['x'].shape
    training_summary['# batches'] = n_batches
    training_summary['Patch size'] = patch_size
    training_summary['Batch size'] = batch_size
    training_summary['Sampling rate'] = sampling_rate
    training_summary['Start epoch'] = start_epoch
    training_summary['Output directory'] = out_directory

    print('\n## Training summary')
    for k, v in training_summary.items():
        print('{:30s}: {}'.format(k, v))
    print('', flush=True)

    with tqdm(total=n_epochs, ncols=TQDM_WIDTH, desc='Train {} for {}'.format(type(model).__name__, camera_name)) as pbar:
        pbar.update(start_epoch)

        for epoch in range(start_epoch, n_epochs):

            for batch_id in range(n_batches):
                batch_x, batch_y = data.next_training_batch(batch_id, batch_size, patch_size, discard_flat=False)
                loss = model.training_step(batch_x, batch_y, learning_rate)
                loss_local.append(loss)

            model.performance['loss']['training'].append(float(np.mean(loss_local)))
            losses_buf.append(model.performance['loss']['training'][-1])

            if epoch == start_epoch:
                developed = np.zeros_like(data['validation']['y'], dtype=np.float32)

            if epoch % sampling_rate == 0:
                # Use the current model to develop images in the validation set
                developed_old = developed
                ssims, psnrs, v_losses, developed = validate(model, data, out_directory, True, epoch, True, loss_metric=model.loss_metric)
                model.performance['ssim']['validation'].append(float(np.mean(ssims)))
                model.performance['psnr']['validation'].append(float(np.mean(psnrs)))
                model.performance['loss']['validation'].append(float(np.mean(v_losses)))

                # Compare the current images to the ones from a previous model iteration
                dmses = []
                for v in range(data['validation']['x'].shape[0]):
                    dmses.append(compare_mse(developed_old[v, :, :, :], developed[v, :, :, :]))

                model.performance['dmse']['validation'].append(np.mean(dmses))

                # Generate progress summary
                training_summary['Epoch'] = epoch
                visualize_progress(model.class_name, model.performance, patch_size, camera_name, out_directory, False, sampling_rate)
                save_progress(model.performance, training_summary, out_directory)
                model.save_model(out_directory, epoch)

                # Check for convergence
                if len(model.performance['loss']['validation']) > 10:
                    current = np.mean(model.performance['loss']['validation'][-n_tail:-1])
                    previous = np.mean(model.performance['loss']['validation'][-(n_tail + 1):-2])
                    vloss_change = abs((current - previous) / previous)

                    if vloss_change < validation_loss_threshold:
                        print('Early stopping - the model converged, validation loss change {}'.format(vloss_change))
                        break

            pbar.set_postfix(loss=np.mean(losses_buf), psnr=model.performance['psnr']['validation'][-1], dmse=np.log10(model.performance['dmse']['validation'][-1]))
            pbar.update(1)

    training_summary['Epoch'] = epoch
    visualize_progress(model.class_name, model.performance, patch_size, camera_name, out_directory, False, sampling_rate)
    save_progress(model.performance, training_summary, out_directory)
    model.save_model(out_directory, epoch)

    return out_directory
