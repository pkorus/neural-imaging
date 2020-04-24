import os
import types
import json
import numpy as np
from matplotlib.figure import Figure

from collections import OrderedDict

import helpers.stats
from helpers import utils, plots, metrics

from models.pipelines import NIPModel
from models.compression import DCN
from models.jpeg import JPEG

from loguru import logger


def validate_jpeg(jpeg, data, batch_size=1):
    
    if not isinstance(jpeg, JPEG):
        raise ValueError('Codec needs to be as instance of {} but is {}'.format(JPEG, jpeg.class_name))

    batch_size = np.minimum(batch_size, data.count_validation)
    n_batches = data.count_validation // batch_size

    results = {k: [] for k in ('psnr', 'ssim', 'entropy')}

    for batch_id in range(n_batches):
        batch_x = data.next_validation_batch(batch_id, batch_size)
        if isinstance(batch_x, tuple):
            batch_x = batch_x[-1]

        batch_y, entropy = jpeg.process(batch_x, return_entropy=True)
        batch_y = batch_y.numpy()

        results['ssim'].append(metrics.batch(batch_x, batch_y, metrics.ssim))
        results['psnr'].append(metrics.batch(batch_x, batch_y, metrics.psnr))
        results['entropy'].append(entropy)

    return {k: float(np.mean(v)) for k, v in results.items()}


def validate_dcn(dcn, data, save_dir=False, epoch=0, show_ref=False):
    """
    Computes validation metrics for a compression model (DCN). (If not a DCN, the function returns immediately).
    If requested, plot compressed images to a JPEG image.

    :param dcn: the DCN model
    :param data: the dataset (instance of Dataset)
    :param data: the dataset (instance of Dataset)
    :param save_dir: path to the directory where figures should be generated
    :param epoch: epoch counter to be appended to the output filename
    :param show_ref: whether to show only the compressed image or also the input image as reference
    :return: tuple of lists with per-image measurements of (ssims, psnrs, losses, entropies)
    """
    
    if not isinstance(dcn, DCN):
        return

    # Compute latent representations and compressed output
    batch_x = data.next_validation_batch(0, data.count_validation)
    if isinstance(batch_x, tuple):
        batch_x = batch_x[-1]
    batch_y, entropy = dcn.process(batch_x, return_entropy=True)
    entropy = float(entropy.numpy())

    ssim = metrics.ssim(batch_x, batch_y.numpy()).tolist()
    psnr = metrics.psnr(batch_x, batch_y.numpy()).tolist()

    loss = float(dcn.loss(batch_x, batch_y, entropy).numpy())

    # If requested, plot a figure with input/output pairs
    if save_dir is not None:
        images_x = np.minimum(data.count_validation, 10 if not show_ref else 5)
        images_y = np.ceil(data.count_validation / images_x)
        fig = Figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))

        for b in range(data.count_validation):
            ax = fig.add_subplot(images_y, images_x, b + 1)
            plots.image(
                np.concatenate((batch_x[b], batch_y[b]), axis=1) if show_ref else batch_y[b],
                '{:.1f} / {:.2f}'.format(psnr[b], ssim[b]),
                axes=ax
            )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig('{}/dcn_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100, quality=90)
        del fig
    
    return {'ssim': float(np.mean(ssim)), 'psnr': float(np.mean(psnr)), 'loss': loss, 'entropy': entropy}


def validate_nip(model, data, save_dir=False, epoch=0, show_ref=False, loss_type='L2'):
    """
    Develops image patches using the given NIP and returns standard image quality measures.
    If requested, resulting patches are visualized as thumbnails and saved to a directory.

    :param model: the NIP model
    :param data: the dataset (instance of Dataset)
    :param data: the dataset (instance of Dataset)
    :param save_dir: path to the directory where figures should be generated
    :param epoch: epoch counter to be appended to the output filename
    :param show_ref: whether to show only the developed image or also the GT target
    :param loss_type: L1 or L2
    :return: tuple of lists with per-image measurements of (ssims, psnrs, losss)
    """

    ssims = []
    psnrs = []
    losss = []

    # If requested, plot a figure with output/target pairs
    if save_dir is not None:
        images_x = np.minimum(data.count_validation, 10 if not show_ref else 5)
        images_y = np.ceil(data.count_validation / images_x)
        fig = Figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))
        
    developed_out = np.zeros_like(data['validation']['y'], dtype=np.float32)

    for b in range(data.count_validation):
        example_x, example_y = data.next_validation_batch(b, 1)
        developed = model.process(example_x).numpy().clip(0, 1)
        developed_out[b, :, :, :] = developed
        developed = developed[:, :, :, :].squeeze()
        reference = example_y.squeeze()

        # Compute stats
        ssim = metrics.ssim(reference, developed).mean()
        psnr = metrics.psnr(reference, developed).mean()
        
        if loss_type == 'L2':
            loss = np.mean(np.power(reference - developed, 2.0))
        elif loss_type == 'L1':
            loss = np.mean(np.abs(reference - developed))
        else:
            raise ValueError('Invalid loss! Use either L1 or L2.')
            
        ssims.append(ssim)
        psnrs.append(psnr)
        losss.append(loss)
        
        # Add images to the plot
        if save_dir is not None:
            ax = fig.add_subplot(images_y, images_x, b+1)
            plots.image(
                np.concatenate((reference, developed), axis=1) if show_ref else developed,
                '{:.1f} dB / {:.2f}'.format(psnr, ssim),
                axes=ax
            )

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/nip_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100, quality=90)
        del fig
    
    return ssims, psnrs, losss


def validate_fan(flow, data, get_labels=False):
    """
    Generates a confusion matrix for the FAN model from a manipulation classification workflow.

    :param flow: manipulation classification workflow
    :param data: the dataset (instance of Dataset)
    :param data: the dataset (instance of Dataset)
    :param get_labels: whether to return the predicted labels
    :return: either the accuracy or tuple (accuracy, predicted labels)
    """

    batch_size = np.minimum(10, data.count_validation)
    n_batches = data.count_validation // batch_size
    n_classes = flow.n_classes
    conf = np.zeros((n_classes, n_classes))    
    out_labels = []
    accuracies = []

    for batch in range(n_batches):

        batch_x = data.next_validation_batch(batch, batch_size)
        if isinstance(batch_x, tuple):
            batch_x = batch_x[0]

        batch_y = flow._batch_labels(len(batch_x))
        predicted_labels = flow.run_workflow_to_decisions(batch_x)

        if get_labels:
            out_labels += [x for x in predicted_labels]

        for c in range(n_classes):
            for c_ in range(n_classes):
                conf[c, c_] += np.sum((batch_y == c) * (predicted_labels == c_))

        accuracies.append(np.mean(predicted_labels == batch_y))

    if out_labels:
        return np.mean(accuracies), conf / (n_batches * batch_size), out_labels
    else:
        return np.mean(accuracies), conf / (n_batches * batch_size)
        

def visualize_manipulation_training(flow, epoch, save_dir=None):
    """
    Visualize progress of manipulation training.

    :param nip: the neural imaging pipeline
    :param fan: the forensic analysis network
    :param dcn: the compression model (e.g., deep compression network)
    :param conf: confusion matrix (see 'confusion()')
    :param epoch: epoch counter to be appended to the output filename
    :param save_dir: path to the directory where figures should be generated (figure handle returned otherwise)
    :param classes: labels for the classes to be used for plotting the confusion matrix
    :return: None (if output to file requested) or figure handle
    """

    # Basic figure setup
    images_x = 3
    images_y = 3 if isinstance(flow.codec, DCN) else 2
    fig = Figure(figsize=(18, 10 / images_x * images_y))
    conf = np.array(flow.fan.performance['confusion'])
        
    # Draw the plots
    ax = fig.add_subplot(images_y, images_x, 1)
    ax.plot(flow.nip.performance['loss']['training'], '.', alpha=0.25)
    ax.plot(helpers.stats.ma_conv(flow.nip.performance['loss']['training'], 0))
    ax.set_ylabel('{} NIP loss'.format(flow.nip.class_name))
    ax.set_title('Loss')

    ax = fig.add_subplot(images_y, images_x, 2)
    ax.plot(flow.nip.performance['psnr']['validation'], '.', alpha=0.25)
    ax.plot(helpers.stats.ma_conv(flow.nip.performance['psnr']['validation'], 0))
    ax.set_ylabel('{} NIP psnr'.format(flow.nip.class_name))
    ax.set_title('PSNR')
    ax.set_ylim([30, 50])

    ax = fig.add_subplot(images_y, images_x, 3)
    ax.plot(flow.nip.performance['ssim']['validation'], '.', alpha=0.25)
    ax.plot(helpers.stats.ma_conv(flow.nip.performance['ssim']['validation'], 0))
    ax.set_ylabel('{} NIP ssim'.format(flow.nip.class_name))
    ax.set_title('SSIM')
    ax.set_ylim([0.8, 1])
    
    ax = fig.add_subplot(images_y, images_x, 4)
    ax.plot(flow.fan.performance['loss']['training'], '.', alpha=0.25)
    ax.plot(helpers.stats.ma_conv(flow.fan.performance['loss']['training'], 0))
    ax.set_ylabel('FAN loss')

    ax = fig.add_subplot(images_y, images_x, 5)
    ax.plot(flow.fan.performance['accuracy']['validation'], '.', alpha=0.25)
    ax.plot(helpers.stats.ma_conv(flow.fan.performance['accuracy']['validation'], 0))
    ax.set_ylabel('FAN accuracy')
    ax.set_ylim([0, 1])

    # The confusion matrix
    ax = fig.add_subplot(images_y, images_x, 6)
    ax.imshow(conf, vmin=0, vmax=1)

    ax.set_xticks(range(flow.n_classes))
    ax.set_xticklabels(flow._forensics_classes, rotation='vertical')
    ax.set_yticks(range(flow.n_classes))
    ax.set_yticklabels(flow._forensics_classes)

    for r in range(flow.n_classes):
        ax.text(r, r, '{:.2f}'.format(conf[r, r]), horizontalalignment='center', color='b' if conf[r, r] > 0.5 else 'w')

    ax.set_xlabel('PREDICTED class')
    ax.set_ylabel('TRUE class')
    ax.set_title('Accuracy: {:.2f}'.format(np.mean(np.diag(conf))))

    # If the compression model is a trainable DCN, include it's validation metrics
    if images_y == 3:
        ax = fig.add_subplot(images_y, images_x, 7)
        ax.plot(flow.codec.performance['loss']['validation'], '.', alpha=0.25)
        ax.plot(helpers.stats.ma_conv(flow.codec.performance['loss']['validation'], 0))
        ax.set_ylabel('DCN loss')

        ax = fig.add_subplot(images_y, images_x, 8)
        ax.plot(flow.codec.performance['ssim']['validation'], '.', alpha=0.25)
        ax.plot(helpers.stats.ma_conv(flow.codec.performance['ssim']['validation'], 0))
        ax.set_ylabel('DCN ssim')
        ax.set_ylim([0.8, 1])

        ax = fig.add_subplot(images_y, images_x, 9)
        ax.plot(flow.codec.performance['entropy']['validation'], '.', alpha=0.25)
        ax.plot(helpers.stats.ma_conv(flow.codec.performance['entropy']['validation'], 0))
        ax.set_ylabel('DCN entropy')

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/manip_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100)
        del fig

    else:
        return fig


def save_training_progress(training_summary, flow, root_dir, quiet=False):
    """
    Saves training progress to a JSON file.

    :param training_summary: dictionary with additional information (e.g., basic training setup)
    :param nip: the neural imaging pipeline
    :param fan: the forensic analysis network
    :param dcn: the compression model (e.g., a deep compression network)
    :param conf: the confusion matrix of the FAN
    :param root_dir: output directory
    """

    # Populate output structures
    training = OrderedDict()
    training['summary'] = training_summary
    training['distribution'] = flow._distribution
    training['manipulations'] = flow._forensics_classes

    # The neural imaging pipeline (NIP
    training['nip'] = OrderedDict()
    training['nip']['model'] = flow.nip.class_name
    training['nip']['init'] = repr(flow.nip)
    training['nip']['args'] = flow.nip._h.to_json() if hasattr(flow.nip, '_h') else {}
    training['nip']['performance'] = flow.nip.performance
    
    # The forensic analysis network (FAN)
    training['forensics'] = OrderedDict()
    training['forensics']['model'] = flow.fan.class_name
    training['forensics']['init'] = repr(flow.fan)
    training['forensics']['args'] = flow.fan._h.to_json()
    training['forensics']['performance'] = flow.fan.performance    
    
    # The deep compression network (DCN)
    if flow.codec is not None:
        training['codec'] = OrderedDict()
        training['codec']['model'] = flow.codec.class_name
        training['codec']['init'] = repr(flow.codec)
    if flow.codec is not None and hasattr(flow.codec, '_h'):
        training['codec']['args'] = flow.codec._h.to_json()
    if flow.codec is not None and hasattr(flow.codec, 'performance'):
        training['codec']['performance'] = flow.codec.performance
    
    # Make dirs if needed
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Save as JSON
    filename = os.path.join(root_dir, 'training.json')
    if not quiet:
        logger.info(f'> Training progress --> {filename}')
    with open(filename, 'w') as f:
        json.dump(training, f, indent=4)
