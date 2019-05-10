import os
import tqdm
import io
import json
import imageio
import numpy as np
import tensorflow as tf

from collections import deque
from skimage.transform import resize, rescale
from skimage.measure import compare_ssim as ssim

import matplotlib.pyplot as plt

# Own libraries and modules
from helpers import plotting, summaries


def visualize_distribution(dcn, data):

    sample_batch_size = np.min((100, data.count_validation))
    batch_x = data.next_validation_batch(0, sample_batch_size)

    # See latent distribution
    batch_z = dcn.compress(batch_x)
    batch_z = batch_z.reshape((sample_batch_size, -1)).T
    
    codebook = dcn.sess.run(dcn.codebook).reshape((-1)).tolist()
    qmin = np.floor(codebook[0])
    qmax = np.ceil(codebook[-1])
    
    feed_dict = {dcn.x: batch_x}
    if hasattr(dcn, 'is_training'):
        feed_dict[dcn.is_training] = True
    
    histogram = dcn.sess.run(dcn.histogram, feed_dict=feed_dict).reshape((-1)).tolist()
        
    # Actual histogram for the quantized latent representation
    bin_centers = np.arange(qmin - 1, qmax + 1, 0.1)
    bin_boundaries = np.convolve(bin_centers, [0.5, 0.5], mode='valid')
    bin_centers = bin_centers[1:-1]
    hist = np.histogram(batch_z[:], bins=bin_boundaries, density=True)[0]
    
    fig = plt.figure(figsize=(10, 2))
    ax = fig.gca()
    ax.set_xlim([qmin - 1, qmax + 1])
    ax.set_xticks(np.arange(qmin, qmax))    
    ax.stem(bin_centers, hist / 10, linefmt='r:', markerfmt='r.') # width=bin_centers[1] - bin_centers[0]
    ax.bar(codebook, histogram, width=(codebook[1] - codebook[0])/2, color='b', alpha=0.5)
    ax.set_title('Histogram of quantized coefficients')
    ax.legend(['Quantized values', 'Soft quantization estimate'])
    
    s = io.BytesIO()
    fig.savefig(s, format='png', bbox_inches='tight')
    plt.close(fig)
    return imageio.imread(s.getvalue(), pilmode='RGB')


def visualize_codebook(dcn):

    qmin = -2 ** (dcn.latent_bpf - 1) + 1
    qmax = 2 ** (dcn.latent_bpf - 1)

    uniform_cbook = np.arange(qmin, qmax + 1)
    codebook = dcn.sess.run(dcn.codebook).reshape((-1)).tolist()

    fig = plt.figure(figsize=(10, 1))

    for x1, x2 in zip(codebook, uniform_cbook):
        fig.gca().plot([x1, x2], [0, 1], 'k:')

    fig.gca().plot(codebook, np.zeros_like(codebook), 'x')
    fig.gca().plot(uniform_cbook, np.ones_like(uniform_cbook), 'ro')
    fig.gca().set_ylim([-1, 2])
    fig.gca().set_xlim([qmin - 1, qmax + 1])
    fig.gca().set_yticks([])
    fig.gca().set_xticks(uniform_cbook)
    s = io.BytesIO()
    fig.savefig(s, format='png', bbox_inches='tight')
    plt.close(fig)
    return imageio.imread(s.getvalue(), pilmode='RGB')


def save_progress(dcn, training, out_dir):

    filename = os.path.join(out_dir, 'progress.json')
    
    output_stats = {
        'training_spec': training,
        'dcn': {
            'model': type(dcn).__name__,
            'args': dcn.args,
            'codebook': dcn.sess.run(dcn.codebook).reshape((-1,)).tolist()
        },
        'performance': dcn.performance,
    }    
    
    with open(filename, 'w') as f:
        json.dump(output_stats, f, indent=4)


def train_dcn(tf_ops, training, data, directory='./data/raw/compression/'):
    """
    tf_ops = {
        'dcn'
    }
    
    training {
    
        'augmentation_probs': {
            'resize': 0.0,
            'flip_h': 0.5,
            'flip_v': 0.5
        }    
    }
    
    """

    dcn = tf_ops['dcn']    
    dcn.init()

    # Compute the number of available batches
    n_batches = data['training']['y'].shape[0] // training['batch_size']
    v_batches = data['validation']['y'].shape[0] // training['batch_size']

    # Structures for storing performance stats
    perf = dcn.performance

    caches = {
        'loss': {'training': deque(maxlen=n_batches), 'validation': deque(maxlen=v_batches)},
        'entropy': {'training': deque(maxlen=n_batches), 'validation': deque(maxlen=v_batches)},
        'ssim': {'training': deque(maxlen=n_batches), 'validation': deque(maxlen=v_batches)}
    }

    n_tail = 3
    learning_rate = training['learning_rate']
    model_output_dirname = os.path.join(directory, dcn.model_code, dcn.scoped_name)

    print('Output directory: {}'.format(model_output_dirname))

    # Create a summary writer and create the necessary directories
    sw = dcn.get_summary_writer(model_output_dirname)

    with tqdm.tqdm(total=training['n_epochs'], ncols=160, desc=dcn.model_code.split('/')[-1]) as pbar:

        for epoch in range(0, training['n_epochs']):
            
            training['current_epoch'] = epoch

            if epoch > 0 and epoch % training['learning_rate_reduction_schedule'] == 0:
                learning_rate *= training['learning_rate_reduction_factor']

            # Iterate through batches of the training data 
            for batch_id in range(n_batches):

                # Pick random patch size - will be resized later for augmentation
                current_patch = np.random.choice(np.arange(training['patch_size'], 2 * training['patch_size']), 1) if np.random.uniform() < training['augmentation_probs']['resize'] else training['patch_size']

                # Sample next batch
                batch_x = data.next_training_batch(batch_id, training['batch_size'], current_patch)

                # If rescaling needed, apply
                if training['patch_size'] != current_patch:
                    batch_t = np.zeros((batch_x.shape[0], training['patch_size'], training['patch_size'], 3), dtype=np.float32)
                    for i in range(len(batch_x)):
                        batch_t[i] = resize(batch_x[i], [training['patch_size'], training['patch_size']], anti_aliasing=True)
                    batch_x = batch_t                

                # Data augmentation - random horizontal flip
                if np.random.uniform() < training['augmentation_probs']['flip_h']: batch_x = batch_x[:, :, ::-1, :]
                if np.random.uniform() < training['augmentation_probs']['flip_v']: batch_x = batch_x[:, ::-1, :, :]

                # Sample dropout
                keep_prob = 1.0 if not training['sample_dropout'] else np.random.uniform(0.5, 1.0)            

                # Make a training step
                values = dcn.training_step(batch_x, learning_rate, dropout_keep_prob=keep_prob)
                
                # TODO temporary nan hook                
                if np.isnan(values['loss']):
                    print('NaN loss detected - dumping current variables')
                    codebook = dcn.sess.run(dcn.codebook).reshape((-1,))
                    # Get some extra stats
                    if dcn.scale_latent:
                        scaling = dcn.sess.run(dcn.graph.get_tensor_by_name('autoencoderdcn/encoder/latent_scaling:0'))
                    else:
                        scaling = np.nan
                    print('Scaling: {}'.format(scaling))
                    print('Codebook: {}'.format(codebook.tolist()))                    
                    # Dump all variables to check which is nan
                    for var in dcn.parameters:
                        if np.any(np.isnan(dcn.sess.run(var))):
                            nan_perc = np.mean(np.isnan(dcn.sess.run(var)))
                            print('!! NaNs found in {} --> {}'.format(var.name, nan_perc))
                    return None
                
                for key, value in values.items():
                    caches[key]['training'].append(value)                

            # Record average values for the whole epoch
            for key in ['loss', 'ssim', 'entropy']:
                perf[key]['training'].append(float(np.mean(caches[key]['training'])))

            # Get some extra stats
            if dcn.scale_latent:
                scaling = dcn.sess.run(dcn.graph.get_tensor_by_name('autoencoderdcn/encoder/latent_scaling:0'))
            else:
                scaling = np.nan

            codebook = dcn.sess.run(dcn.codebook).reshape((-1,))

            # Iterate through batches of the validation data
            if epoch % training['validation_schedule'] == 0:

                for batch_id in range(v_batches):
                    batch_x = data.next_validation_batch(batch_id, training['batch_size'])
                    batch_y = dcn.process(batch_x, is_training=training['validation_is_training'])

                    # Compute loss
                    loss_value = np.linalg.norm(batch_x - batch_y)
                    caches['loss']['validation'].append(loss_value)                

                    # Compute SSIM
                    ssim_value = np.mean([ssim(batch_x[r], batch_y[r], multichannel=True) for r in range(len(batch_x))]) 
                    caches['ssim']['validation'].append(ssim_value)

                perf['loss']['validation'].append(float(np.mean(caches['loss']['validation'])))
                perf['ssim']['validation'].append(float(np.mean(caches['ssim']['validation'])))

                # Save current snapshot
                thumbs = (255 * plotting.thumbnails(np.concatenate((batch_x[::2], batch_y[::2]), axis=0), n_cols=20)).astype(np.uint8)
                thumbs_few = (255 * plotting.thumbnails(np.concatenate((batch_x[::10], batch_y[::10]), axis=0), n_cols=4)).astype(np.uint8)
                imageio.imsave(os.path.join(model_output_dirname, 'thumbnails-{:05d}.png'.format(epoch)), thumbs)

                # Sample latent space
                batch_z = dcn.compress(batch_x)

                # Save summaries to TB            
                summary = tf.Summary()
                summary.value.add(tag='loss/validation', simple_value=perf['loss']['validation'][-1])
                summary.value.add(tag='loss/training', simple_value=perf['loss']['training'][-1])
                summary.value.add(tag='ssim/validation', simple_value=perf['ssim']['validation'][-1])
                summary.value.add(tag='ssim/training', simple_value=perf['ssim']['training'][-1])
                summary.value.add(tag='entropy/training', simple_value=perf['entropy']['training'][-1])
                summary.value.add(tag='scaling', simple_value=scaling)
                summary.value.add(tag='images/reconstructed', image=summaries.log_image(rescale(thumbs_few, 1.0, anti_aliasing=True)))
                summary.value.add(tag='histograms/latent', histo=summaries.log_histogram(batch_z))
                summary.value.add(tag='histograms/latent_approx', image=summaries.log_image(visualize_distribution(dcn, data)))
                
                if dcn.train_codebook:
                    summary.value.add(tag='codebook/min', simple_value=codebook.min())
                    summary.value.add(tag='codebook/max', simple_value=codebook.max())
                    summary.value.add(tag='codebook/mean', simple_value=codebook.mean())
                    summary.value.add(tag='codebook/diff_variance', simple_value=np.var(np.convolve(codebook, [-1, 1], mode='valid')))                    
                    summary.value.add(tag='codebook/centroids', image=summaries.log_image(visualize_codebook(dcn)))

                sw.add_summary(summary, epoch)
                sw.flush()
                
                # Save stats to a JSON log
                save_progress(dcn, training, model_output_dirname)                                
                
                # Save current checkpoint
                dcn.save_model(model_output_dirname, epoch)

                # Check for convergence or model deterioration
                if len(perf['ssim']['validation']) > 5:
                    current = np.mean(perf['ssim']['validation'][-n_tail:-1])
                    previous = np.mean(perf['ssim']['validation'][-(n_tail + 1):-2])
                    perf_change = abs((current - previous) / previous)

                    if perf_change < training['convergence_threshold']:
                        print('Early stopping - the model converged, validation SSIM change {}'.format(perf_change))
                        break

                    if current < previous:
                        print('Early stopping - SSIM deterioration {} -> {}'.format(previous, current))
                        break

            progress_dict = {
                'L': np.mean(perf['loss']['training'][-3:]),
                'Lv': np.mean(perf['loss']['validation'][-1:]),
                'lr': '{:.1e}'.format(learning_rate),
                'ssim': '{:.2f}'.format(perf['ssim']['validation'][-1]),
                'H': '{:.1f}'.format(np.mean(perf['entropy']['training'][-1:])),                
                # 'Qvar': np.var(np.convolve(codebook, [-1, 1], mode='valid')),
            }
            
            if dcn.scale_latent:
                progress_dict['S'] = '{:.1f}'.format(scaling)

            if dcn.use_batchnorm:
                # Get current batch / population stats
                prebn = dcn.sess.run(dcn.pre_bn, feed_dict={dcn.x: batch_x})
                bM = np.mean(prebn, axis=(0, 1, 2))
                bV = np.var(prebn, axis=(0, 1, 2))
                pM = dcn.sess.run(dcn.graph.get_tensor_by_name('autoencoderdcn/encoder/bn_0/moving_mean:0'))
                pV = dcn.sess.run(dcn.graph.get_tensor_by_name('autoencoderdcn/encoder/bn_0/moving_variance:0'))
                # Append summary
                progress_dict['MVp'] = '{:.2f}/{:.2f}'.format(np.mean(pM), np.mean(pV))
                progress_dict['MVb'] = '{:.2f}/{:.2f}'.format(np.mean(bM), np.mean(bV))

            # Update progress bar
            pbar.set_postfix(progress_dict)
            pbar.update(1)
