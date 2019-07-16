import os
import types
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from helpers import utils
from skimage.measure import compare_ssim, compare_psnr, compare_mse
from models.pipelines import NIPModel
from models.compression import DCN


def confusion(mc, data, lagel_generator, label_multiplier=1):
    """ Generates a confusion matrix for a FAN model."""

    batch_size = np.minimum(10, data.count_validation)
    n_batches = data.count_validation // batch_size
    n_classes = mc.n_classes
    
    conf = np.zeros((n_classes, n_classes))
    
    for batch in range(n_batches):
        batch_x, _ = data.next_validation_batch(batch, batch_size)

        if type(lagel_generator) is types.FunctionType:
            batch_y = lagel_generator(len(batch_x))
        else:
            batch_y = lagel_generator[(batch*batch_size*label_multiplier):(batch+1)*batch_size*label_multiplier]
        predicted_labels = mc.process(batch_x)
        
        for c in range(n_classes):
            for c_ in range(n_classes):
                conf[c, c_] += np.sum( (batch_y == c) * (predicted_labels == c_))

    return conf / data.count_validation


def validate_dcn(model, data, save_dir=False, epoch=0, show_ref=False):
    
    if not isinstance(model, DCN):
        return

    if save_dir is not None:
        # Setup output figure
        images_x = np.minimum(data.count_validation, 10 if not show_ref else 5)
        images_y = np.ceil(data.count_validation / images_x)
        fig = plt.figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))
        
    developed_out = np.zeros_like(data['validation']['y'], dtype=np.float32)

    # Entropy
    batch_x = data.next_validation_batch(0, data.count_validation)[-1]
    batch_z = model.compress(batch_x, direct=True)
    batch_y = model.process(batch_x, direct=True)
    
    codebook = model.get_codebook()
    
    ssims = [compare_ssim(batch_x[b], batch_y[b], multichannel=True, data_range=1) for b in range(data.count_validation)]
    mses = [compare_mse(batch_x[b], batch_y[b]) for b in range(data.count_validation)]
    entropies = [utils.entropy(batch_z[b], codebook) for b in range(data.count_validation)]
    
    if save_dir is not None:
        for b in range(data.count_validation):
            ax = fig.add_subplot(images_y, images_x, b+1)
            if show_ref:
                ax.imshow(np.concatenate( (batch_x[b], batch_y[b]), axis=1))
            else:
                ax.imshow(batch_y[b])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{:.1f} / {:.2f}'.format(mses[b], ssims[b]), fontsize=6)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/dcn_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=150)
        plt.close(fig)
        del fig
    
    return ssims, mses, entropies

def validate_nip(model, data, save_dir=False, epoch=0, show_ref=False, loss_type='L2'):
    """ Develops image patches using the given NIP and returns standard image quality measures.
        If requested, resulting patches are visualized as thumbnails and saved to a directory.
    """

    ssims = []
    psnrs = []
    losss = []

    if save_dir is not None:
        # Setup output figure
        images_x = np.minimum(data.count_validation, 10 if not show_ref else 5)
        images_y = np.ceil(data.count_validation / images_x)
        fig = plt.figure(figsize=(20, 20 / images_x * images_y * (1 if not show_ref else 0.5)))
        
    developed_out = np.zeros_like(data['validation']['y'], dtype=np.float32)

    for b in range(data.count_validation):
        example_x, example_y = data.next_validation_batch(b, 1)
        developed = model.process(example_x)
        developed = np.clip(developed, 0, 1)
        developed_out[b, :, :, :] = developed
        developed = developed[:, :, :, :].squeeze()
        reference = example_y.squeeze()

        # Compute stats
        ssim = compare_ssim(reference, developed, multichannel=True)
        psnr = compare_psnr(reference, developed)
        
        if loss_type == 'L2':
            loss = np.mean(np.power(reference - developed, 2.0))
        elif loss_type == 'L1':
            loss = np.mean(np.abs(reference - developed))
        else:
            raise ValueError('Invalid loss type!')
            
        ssims.append(ssim)
        psnrs.append(psnr)
        losss.append(loss)
        
        # Display validation results
        if save_dir is not None:
            ax = fig.add_subplot(images_y, images_x, b+1)
            if show_ref:
                ax.imshow(np.concatenate( (reference, developed), axis=1))
            else:
                ax.imshow(developed)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{:.1f} dB / {:.2f}'.format(psnr, ssim), fontsize=6)

    if save_dir is not None:
        # # Save the figure
        # if '{' in save_dir and '}' in save_dir:
        #     dirname = save_dir.replace('{nip-model}', type(model).__name__)
        # else:
        # dirname = os.path.join(save_dir, type(model).__name__)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/nip_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=150)
        plt.close(fig)
        del fig
    
    return ssims, psnrs, losss


def validate_fan(mc, data, label_generator, label_multiplier, get_labels=False):
    """ Computes the average accuracy for a forensics network. """

    batch_size = np.minimum(10, data.count_validation)
    n_batches = data.count_validation // batch_size
    out_labels = []
    accurracies = []

    for batch in range(n_batches):

        batch_x, _ = data.next_validation_batch(batch, batch_size)

        if type(label_generator) is types.FunctionType:
            batch_y = label_generator(len(batch_x))
        else:
            batch_y = label_generator[(batch * batch_size * label_multiplier):(batch + 1) * batch_size * label_multiplier]

        if label_multiplier * len(batch_x) != len(batch_y):
            raise RuntimeError('Number of labels is not equal to the number of examples! {} x {} vs. {}'.format(label_multiplier, len(batch_x), len(batch_y)))
        
        predicted_labels = mc.process(batch_x)
        if get_labels:
            out_labels += [x for x in predicted_labels]

        accurracies.append(np.mean(predicted_labels == batch_y))
    if out_labels:
        return np.mean(accurracies), out_labels
    else:
        return np.mean(accurracies)
        

def visualize_manipulation_training(nip, fornet, conf, epoch, save_dir=None, classes=None):
    """ Visualizes progress of manipulation detection training. """
        
    # Init
    images_x = 3
    images_y = 2
    fig = plt.figure(figsize=(18, 10 / images_x * images_y))
        
    # Draw the plots
    ax = fig.add_subplot(images_y, images_x, 1)
    ax.plot(nip.train_perf['loss'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(nip.train_perf['loss'], 0))
    ax.set_ylabel('{} NIP loss'.format(type(nip).__name__))
    ax.set_title('Loss')

    ax = fig.add_subplot(images_y, images_x, 2)
    ax.plot(nip.valid_perf['psnr'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(nip.valid_perf['psnr'], 0))
    ax.set_ylabel('{} NIP psnr'.format(type(nip).__name__))
    ax.set_title('PSNR')
    ax.set_ylim([30, 50])

    ax = fig.add_subplot(images_y, images_x, 3)
    ax.plot(nip.valid_perf['ssim'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(nip.valid_perf['ssim'], 0))
    ax.set_ylabel('{} NIP psnr'.format(type(nip).__name__))
    ax.set_title('SSIM')
    ax.set_ylim([0.8, 1])
    
    ax = fig.add_subplot(images_y, images_x, 4)
    ax.plot(fornet.train_perf['loss'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(fornet.train_perf['loss'], 0))
    ax.set_ylabel('Forensics network\'s loss')

    ax = fig.add_subplot(images_y, images_x, 5)
    ax.plot(fornet.valid_perf['accuracy'], '.', alpha=0.25)
    ax.plot(utils.ma_conv(fornet.valid_perf['accuracy'], 0))
    ax.set_ylabel('Forensics network\'s accuracy')
    ax.set_ylim([0, 1])
        
    ax = fig.add_subplot(images_y, images_x, 6)
    ax.imshow(conf, vmin=0, vmax=1)
    if classes is not None:
        ax.set_xticks(range(fornet.n_classes))
        ax.set_xticklabels(classes, rotation='vertical')
        ax.set_yticks(range(fornet.n_classes))
        ax.set_yticklabels(classes)
    for r in range(fornet.n_classes):
        ax.text(r, r, '{:.2f}'.format(conf[r, r]), horizontalalignment='center', color='b' if conf[r,r] > 0.5 else 'w')        
    ax.set_xlabel('PREDICTED class')
    ax.set_ylabel('TRUE class')
    ax.set_title('Accuracy: {:.2f}'.format(np.mean(np.diag(conf))))

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig.savefig('{}/manip_validation_{:05d}.jpg'.format(save_dir, epoch), bbox_inches='tight', dpi=100)
        plt.close(fig)    
        del fig


def save_training_progress(training_summary, model, fan, dcn, conf, root_dir):
    """ Saves training progress to a JSON file."""
    
    # Populate output structures
    training = OrderedDict()
    training['summary'] = training_summary
    
    if isinstance(model, NIPModel):    
        training['nip'] = OrderedDict()
        training['nip']['training'] = OrderedDict() 
        training['nip']['training']['loss'] = model.train_perf['loss']    
        training['nip']['validation'] = OrderedDict() 
        training['nip']['validation']['ssim'] = model.valid_perf['ssim']
        training['nip']['validation']['ssim'] = model.valid_perf['ssim']
        training['nip']['validation']['psnr'] = model.valid_perf['psnr']
        training['nip']['validation']['loss'] = model.valid_perf['loss']
    elif hasattr(model, '__getitem__'):
        for m in model:
            name = 'nip/{}'.format(m.name)
            training[name] = OrderedDict()
            training[name]['training'] = OrderedDict() 
            training[name]['training']['loss'] = m.train_perf['loss']    
            training[name]['validation'] = OrderedDict() 
            training[name]['validation']['ssim'] = m.valid_perf['ssim']
            training[name]['validation']['ssim'] = m.valid_perf['ssim']
            training[name]['validation']['psnr'] = m.valid_perf['psnr']
            training[name]['validation']['loss'] = m.valid_perf['loss']            
    else:
        raise ValueError('Unsupported value passed as a NIP model ({})'.format(type(model)))
    
    if fan is not None:
        training['forensics'] = OrderedDict()
        training['forensics']['training'] = OrderedDict()
        training['forensics']['training']['loss'] = fan.train_perf['loss']
        training['forensics']['validation'] = OrderedDict() 
        training['forensics']['validation']['accuracy'] = fan.valid_perf['accuracy']
    
        if conf is not None:
            training['forensics']['validation']['confusion'] = conf.tolist()
    
    if dcn is not None and hasattr(dcn, 'performance'):
        training['compression'] = OrderedDict()
#         training['compression']['training'] = OrderedDict()
#         training['compression']['training']['ssim'] = dcn.train_perf['loss']
        training['compression']['validation'] = OrderedDict() 
        training['compression']['validation']['ssim'] = dcn.performance['ssim']['validation']
        training['compression']['validation']['entropy'] = dcn.performance['entropy']['validation']
        training['compression']['validation']['loss'] = dcn.performance['loss']['validation']
    
    # Make dirs if needed
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Save as JSON
    with open(os.path.join(root_dir, 'training.json'), 'w') as f:
        json.dump(training, f, indent=4)
