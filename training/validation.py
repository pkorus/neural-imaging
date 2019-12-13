import os
import types
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from helpers import utils, plotting
from skimage.measure import compare_ssim, compare_psnr
from models.pipelines import NIPModel
from models.compression import DCN


def confusion(fan, data, label_generator, label_multiplier=1):
    """
    Generates a confusion matrix for the FAN model on the validation set.

    :param fan: FAN model
    :param data: the dataset (instance of IPDataset)
    :param label_generator: a lambda function which generates GT class labels given the size of the batch
    :param label_multiplier:
    :return: 2-D numpy array with the confusion matrix
    """

    batch_size = np.minimum(10, data.count_validation)
    n_batches = data.count_validation // batch_size
    n_classes = fan.n_classes
    
    conf = np.zeros((n_classes, n_classes))
    
    for batch in range(n_batches):

        batch_x = data.next_validation_batch(batch, batch_size)
        if isinstance(batch_x, tuple):
            batch_x = batch_x[0]

        if isinstance(label_generator, types.FunctionType):
            batch_y = label_generator(len(batch_x))
        else:
            batch_y = label_generator[(batch * batch_size * label_multiplier):(batch + 1) * batch_size * label_multiplier]

        predicted_labels = fan.process(batch_x)

        for c in range(n_classes):
            for c_ in range(n_classes):
                conf[c, c_] += np.sum((batch_y == c) * (predicted_labels == c_))

    if n_batches * batch_size != data.count_validation:
        print('WARNING Not enough images to fill the last batch ({})'.format(batch_size))

    return conf / (n_batches * batch_size)


def validate_dcn(dcn, data, save_dir=False, epoch=0, show_ref=False):
    """
    Computes validation metrics for a compression model (DCN). (If not a DCN, the function returns immediately).
    If requested, plot compressed images to a JPEG image.

    :param dcn: the DCN model
    :param data: the dataset (instance of IPDataset)
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
    batch_z = dcn.compress(batch_x, direct=True)
    batch_y = dcn.process(batch_x, direct=True)

    # Compute quality measures and entropy statistics
    codebook = dcn.get_codebook()
    
    ssims = [compare_ssim(batch_x[b], batch_y[b], multichannel=True, data_range=1) for b in range(data.count_validation)]
    psnrs = [compare_psnr(batch_x[b], batch_y[b], data_range=1) for b in range(data.count_validation)]
    losses = [dcn.sess.run(dcn.loss, feed_dict={dcn.x: batch_x[b:b + 1]}) for b in range(data.count_validation)]
    entropies = [utils.entropy(batch_z[b], codebook) for b in range(data.count_validation)]

    # If requested, plot a figure with input/output pairs
    if save_dir is not None:
        images_x = np.minimum(data.count_validation, 10 if not show_ref else 5)
        images_y = np.ceil(data.count_validation / images_x)
        fig = plt.figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))

        for b in range(data.count_validation):
            ax = fig.add_subplot(images_y, images_x, b + 1)
            plotting.quickshow(
                np.concatenate((batch_x[b], batch_y[b]), axis=1) if show_ref else batch_y[b],
                '{:.1f} / {:.2f}'.format(psnrs[b], ssims[b]),
                axes=ax
            )

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig('{}/dcn_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100, quality=90)
        plt.close(fig)
        del fig
    
    return ssims, psnrs, losses, entropies


def validate_nip(model, data, save_dir=False, epoch=0, show_ref=False, loss_type='L2'):
    """
    Develops image patches using the given NIP and returns standard image quality measures.
    If requested, resulting patches are visualized as thumbnails and saved to a directory.

    :param model: the NIP model
    :param data: the dataset (instance of IPDataset)
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
        fig = plt.figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))
        
    developed_out = np.zeros_like(data['validation']['y'], dtype=np.float32)

    for b in range(data.count_validation):
        example_x, example_y = data.next_validation_batch(b, 1)
        developed = model.process(example_x).clip(0, 1)
        developed_out[b, :, :, :] = developed
        developed = developed[:, :, :, :].squeeze()
        reference = example_y.squeeze()

        # Compute stats
        ssim = compare_ssim(reference, developed, multichannel=True, data_range=1)
        psnr = compare_psnr(reference, developed, data_range=1)
        
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
            plotting.quickshow(
                np.concatenate((reference, developed), axis=1) if show_ref else developed,
                '{:.1f} dB / {:.2f}'.format(psnr, ssim),
                axes=ax
            )

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/nip_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100, quality=90)
        plt.close(fig)
        del fig
    
    return ssims, psnrs, losss


def validate_fan(fan, data, label_generator, label_multiplier, get_labels=False):
    """
    Generates a confusion matrix for the FAN model on the validation set.

    :param fan: FAN model
    :param data: the dataset (instance of IPDataset)
    :param label_generator: a lambda function which generates GT class labels given the size of the batch
    :param label_multiplier:
    :param get_labels: whether to return the predicted labels
    :return: either the accuracy or tuple (accuracy, predicted labels)
    """

    batch_size = np.minimum(10, data.count_validation)
    n_batches = data.count_validation // batch_size
    out_labels = []
    accuracies = []

    for batch in range(n_batches):

        batch_x = data.next_validation_batch(batch, batch_size)
        if isinstance(batch_x, tuple):
            batch_x = batch_x[0]

        if isinstance(label_generator, types.FunctionType):
            batch_y = label_generator(len(batch_x))
        else:
            batch_y = label_generator[(batch * batch_size * label_multiplier):(batch + 1) * batch_size * label_multiplier]

        if label_multiplier * len(batch_x) != len(batch_y):
            raise RuntimeError('Number of labels is not equal to the number of examples! {} x {} vs. {}'.format(label_multiplier, len(batch_x), len(batch_y)))

        predicted_labels = fan.process(batch_x)

        if get_labels:
            out_labels += [x for x in predicted_labels]

        accuracies.append(np.mean(predicted_labels == batch_y))

    if out_labels:
        return np.mean(accuracies), out_labels
    else:
        return np.mean(accuracies)
        

def visualize_manipulation_training(nip, fan, dcn, conf, epoch, save_dir=None, classes=None):
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
    images_y = 3 if isinstance(dcn, DCN) else 2
    fig = plt.figure(figsize=(18, 10 / images_x * images_y))
        
    # Draw the plots
    ax = fig.add_subplot(images_y, images_x, 1)
    ax.plot(nip.performance['loss']['training'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(nip.performance['loss']['training'], 0))
    ax.set_ylabel('{} NIP loss'.format(type(nip).__name__))
    ax.set_title('Loss')

    ax = fig.add_subplot(images_y, images_x, 2)
    ax.plot(nip.performance['psnr']['validation'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(nip.performance['psnr']['validation'], 0))
    ax.set_ylabel('{} NIP psnr'.format(type(nip).__name__))
    ax.set_title('PSNR')
    ax.set_ylim([30, 50])

    ax = fig.add_subplot(images_y, images_x, 3)
    ax.plot(nip.performance['ssim']['validation'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(nip.performance['ssim']['validation'], 0))
    ax.set_ylabel('{} NIP ssim'.format(type(nip).__name__))
    ax.set_title('SSIM')
    ax.set_ylim([0.8, 1])
    
    ax = fig.add_subplot(images_y, images_x, 4)
    ax.plot(fan.performance['loss']['training'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(fan.performance['loss']['training'], 0))
    ax.set_ylabel('FAN loss')

    ax = fig.add_subplot(images_y, images_x, 5)
    ax.plot(fan.performance['accuracy']['validation'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(fan.performance['accuracy']['validation'], 0))
    ax.set_ylabel('FAN accuracy')
    ax.set_ylim([0, 1])

    # The confusion matrix
    ax = fig.add_subplot(images_y, images_x, 6)
    ax.imshow(conf, vmin=0, vmax=1)

    if classes is not None:
        ax.set_xticks(range(fan.n_classes))
        ax.set_xticklabels(classes, rotation='vertical')
        ax.set_yticks(range(fan.n_classes))
        ax.set_yticklabels(classes)

    for r in range(fan.n_classes):
        ax.text(r, r, '{:.2f}'.format(conf[r, r]), horizontalalignment='center', color='b' if conf[r, r] > 0.5 else 'w')

    ax.set_xlabel('PREDICTED class')
    ax.set_ylabel('TRUE class')
    ax.set_title('Accuracy: {:.2f}'.format(np.mean(np.diag(conf))))

    # If the compression model is a trainable DCN, include it's validation metrics
    if images_y == 3:
        ax = fig.add_subplot(images_y, images_x, 7)
        ax.plot(dcn.performance['loss']['validation'], '.', alpha=0.25)
        ax.plot(utils.ma_conv(dcn.performance['loss']['validation'], 0))
        ax.set_ylabel('DCN loss')

        ax = fig.add_subplot(images_y, images_x, 8)
        ax.plot(dcn.performance['ssim']['validation'], '.', alpha=0.25)
        ax.plot(utils.ma_conv(dcn.performance['ssim']['validation'], 0))
        ax.set_ylabel('DCN ssim')
        ax.set_ylim([0.8, 1])

        ax = fig.add_subplot(images_y, images_x, 9)
        ax.plot(dcn.performance['entropy']['validation'], '.', alpha=0.25)
        ax.plot(utils.ma_conv(dcn.performance['entropy']['validation'], 0))
        ax.set_ylabel('DCN entropy')

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/manip_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100)
        plt.close(fig)    
        del fig

    else:
        return fig


def save_training_progress(training_summary, nip, fan, dcn, conf, root_dir):
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

    # The neural imaging pipeline (NIP
    if isinstance(nip, NIPModel):
        training['nip'] = OrderedDict()
        training['nip']['training'] = OrderedDict() 
        training['nip']['training']['loss'] = nip.performance['loss']['training']
        training['nip']['validation'] = OrderedDict() 
        training['nip']['validation']['ssim'] = nip.performance['ssim']['validation']
        training['nip']['validation']['ssim'] = nip.performance['ssim']['validation']
        training['nip']['validation']['psnr'] = nip.performance['psnr']['validation']
        training['nip']['validation']['loss'] = nip.performance['loss']['validation']

    elif hasattr(nip, '__getitem__'):
        for m in nip:
            name = 'nip/{}'.format(m.name)
            training[name] = OrderedDict()
            training[name]['training'] = OrderedDict() 
            training[name]['training']['loss'] = m.performance['loss']['training']
            training[name]['validation'] = OrderedDict() 
            training[name]['validation']['ssim'] = m.performance['ssim']['validation']
            training[name]['validation']['ssim'] = m.performance['ssim']['validation']
            training[name]['validation']['psnr'] = m.performance['psnr']['validation']
            training[name]['validation']['loss'] = m.performance['loss']['validation']
    else:
        raise ValueError('Unsupported value passed as a NIP model ({})'.format(type(nip)))

    # The forensic analysis network (FAN)
    if fan is not None:
        training['forensics'] = OrderedDict()
        training['forensics']['training'] = OrderedDict()
        training['forensics']['training']['loss'] = fan.performance['loss']['training']
        training['forensics']['validation'] = OrderedDict() 
        training['forensics']['validation']['accuracy'] = fan.performance['accuracy']['validation']
    
        if conf is not None:
            training['forensics']['validation']['confusion'] = conf.tolist()

    # The deep compression network (DCN)
    if dcn is not None and hasattr(dcn, 'performance'):
        training['compression'] = OrderedDict()
#         training['compression']['training'] = OrderedDict()
#         training['compression']['training']['ssim'] = dcn.performance['loss']['training']
        training['compression']['validation'] = OrderedDict() 
        training['compression']['validation']['ssim'] = dcn.performance['ssim']['validation']
        training['compression']['validation']['psnr'] = dcn.performance['psnr']['validation']
        training['compression']['validation']['entropy'] = dcn.performance['entropy']['validation']
        training['compression']['validation']['loss'] = dcn.performance['loss']['validation']
    
    # Make dirs if needed
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Save as JSON
    with open(os.path.join(root_dir, 'training.json'), 'w') as f:
        json.dump(training, f, indent=4)
