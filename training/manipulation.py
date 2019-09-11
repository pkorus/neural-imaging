#!/usr/bin/env python
# coding: utf-8

# Basic imports
import os
import numpy as np
import tqdm
from collections import deque, OrderedDict

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# Load my TF models
from models import pipelines
from models.forensics import FAN
from models.jpeg import DJPG
from models import compression

from compression import afi

# Helper functions
from helpers import coreutils, tf_helpers
from training import validation


SUPPORTED_MANIPULATIONS = ['sharpen', 'gaussian', 'jpeg', 'resample', 'awgn', 'gamma', 'median']


@coreutils.logCall
def construct_models(nip_model, patch_size=128, trainable=None, distribution=None, manipulations=None, loss_metric='L2'):
    """
    Setup the TF model of the entire acquisition and distribution workflow.
    :param nip_model: name of the NIP class
    :param patch_size: patch size for manipulation training (raw patch - rgb patches will be 4 times as big)
    :param distribution: definition of the dissemination channel (set to None for the default down+jpeg(50))
    :param loss_metric: NIP loss metric: L2, L1 or SSIM
    """
    # Sanitize inputs
    if patch_size < 32 or patch_size > 512:
        raise ValueError('The patch size ({}) looks incorrect, typical values should be >= 32 and <= 512'.format(patch_size))

    trainable = trainable or {}
    
    # Setup a default distribution channel
    if distribution is None:
        distribution = {
            'downsampling': 'pool',
            'compression': 'jpeg',
            'compression_params': {
                'quality': 50,
                'rounding_approximation': 'sin'
            }
        }
    
    if 'dcn' in trainable and distribution['compression'] != 'dcn':
        raise ValueError('Cannot make DCN trainable given current compression model: {}'.format(distribution['compression']))

    if distribution['compression'] == 'jpeg' and (distribution['compression_params']['quality'] < 1 or distribution['compression_params']['quality'] > 100):
        raise ValueError('Invalid JPEG quality level ({})'.format(distribution['compression_params']['quality']))

    if not issubclass(getattr(pipelines, nip_model), pipelines.NIPModel):
        supported_nips = [x for x in dir(pipelines) if x != 'NIPModel' and type(getattr(pipelines, x)) is type and issubclass(getattr(pipelines, x), pipelines.NIPModel)]
        raise ValueError('Invalid NIP model ({})! Available NIPs: ({})'.format(nip_model, supported_nips))
        
    if loss_metric not in ['L2', 'L1', 'SSIM']:
        raise ValueError('Invalid loss metric ({})!'.format(loss_metric))
    
    tf.reset_default_graph()
    sess = tf.Session()

    # The pipeline -----------------------------------------------------------------------------------------------------

    model = getattr(pipelines, nip_model)(sess, tf.get_default_graph(), patch_size=patch_size, loss_metric=loss_metric)
    print('NIP network: {}'.format(model.summary()))

    # Several paths for post-processing --------------------------------------------------------------------------------
    with tf.name_scope('distribution'):

        manipulations = manipulations or ['sharpen', 'resample', 'gaussian', 'jpeg']

        if any(x not in SUPPORTED_MANIPULATIONS for x in manipulations):
            raise ValueError('Unsupported manipulation requested! Available: {}'.format(SUPPORTED_MANIPULATIONS))

        operations = [model.y]
        forensics_classes = ['native']
        # Sharpen
        if 'sharpen' in manipulations:
            im_shr = tf_helpers.manipulation_sharpen(model.y, 0, hsv=True)
            operations.append(im_shr)
            forensics_classes.append('sharpen')

        # Bilinear resampling
        if 'resample' in manipulations:
            im_res = tf_helpers.manipulation_resample(model.y)
            operations.append(im_res)
            forensics_classes.append('resample')

        # Gaussian filter
        if 'gaussian' in manipulations:
            im_gauss = tf_helpers.manipulation_gaussian(model.y, 5, 0.85)
            operations.append(im_gauss)
            forensics_classes.append('gaussian')

        # Mild JPEG
        if 'jpeg' in manipulations:
            tf_jpg = DJPG(sess, tf.get_default_graph(), model.y, None, quality=80, rounding_approximation='soft')
            operations.append(tf_jpg.y)
            forensics_classes.append('jpeg')

        # AWGN
        if 'awgn' in manipulations:
            im_awgn = tf_helpers.manipulation_awgn(model.y, 0.02)
            operations.append(im_awgn)
            forensics_classes.append('awgn')

        # Gamma + inverse
        if 'gamma' in manipulations:
            im_gamma = tf_helpers.manipulation_gamma(model.y, 3)
            operations.append(im_gamma)
            forensics_classes.append('gamma')

        # Median
        if 'median' in manipulations:
            im_median = tf_helpers.manipulation_median(model.y, 3)
            operations.append(im_median)
            forensics_classes.append('median')

        n_classes = len(operations)
        assert len(forensics_classes) == n_classes

        # Concatenate outputs from multiple post-processing paths ------------------------------------------------------
        y_concat = tf.concat(operations, axis=0)

        # Add sub-sampling and lossy compression in the channel --------------------------------------------------------
        down_patch_size = 2 * patch_size if distribution['downsampling'] == 'none' else patch_size
        if distribution['downsampling'] == 'pool':
            imb_down = tf.nn.avg_pool(y_concat, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='post_downsample')
        elif distribution['downsampling'] == 'bilin':
            imb_down = tf.image.resize_images(y_concat, [tf.shape(y_concat)[1] // 2, tf.shape(y_concat)[1] // 2])
        elif distribution['downsampling'] == 'none':
            imb_down = y_concat
        else:
            raise ValueError('Unsupported channel down-sampling {}'.format(distribution['downsampling']))

        if distribution['compression'] == 'jpeg':

            print('Channel compression: JPEG({quality}, {rounding_approximation})'.format(**distribution['compression_params']))
            dist_compression = DJPG(sess, tf.get_default_graph(), imb_down, model.x, **distribution['compression_params'])
            imb_out = dist_compression.y

        elif distribution['compression'] == 'dcn':

            print('Channel compression: DCN from {dirname}'.format(**distribution['compression_params']))
            if 'dirname' in distribution['compression_params']:
                model_directory = distribution['compression_params']['dirname']
                dist_compression = afi.restore_model(model_directory, down_patch_size, sess=sess, graph=tf.get_default_graph(), x=imb_down, nip_input=model.x)
            else:
                # TODO Not tested yet
                raise NotImplementedError('DCN models should be restored from a pre-training session!')
                # dist_compression = compression.TwitterDCN(sess, tf.get_default_graph(), x=imb_down, nip_input=model.x, patch_size=down_patch_size, **distribution['compression_params'])

            imb_out = dist_compression.y

        elif distribution['compression'] == 'none':
            dist_compression = None
            imb_out = imb_down
        else:
            raise ValueError('Unsupported channel compression {}'.format(distribution['compression']))

    # Add manipulation detection
    fan = FAN(sess, tf.get_default_graph(), n_classes=n_classes, x=imb_out, nip_input=model.x, n_convolutions=4)
    print('Forensics network parameters: {:,}'.format(fan.count_parameters()))

    # Setup a combined loss and training op
    with tf.name_scope('combined_optimization') as scope:
        lambda_nip = tf.placeholder(tf.float32, name='lambda_nip')
        lambda_dcn = tf.placeholder(tf.float32, name='lambda_dcn')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        loss = fan.loss
        if 'nip' in trainable:
            loss += lambda_nip * model.loss
        if 'dcn' in trainable:
            loss += lambda_dcn * dist_compression.loss

        adam = tf.train.AdamOptimizer(learning_rate=lr, name='adam')

        # List parameters that need to be optimized
        parameters = []
        parameters.extend(fan.parameters)
        if 'nip' in trainable:
            parameters.extend(model.parameters)
        if 'dcn' in trainable:
            parameters.extend(dist_compression.parameters)
        opt = adam.minimize(loss, name='opt_combined', var_list=parameters)
        
    tf_ops = {
        'sess': sess,
        'nip': model,
        'fan': fan,
        'loss': loss,
        'opt': opt,
        'lr': lr,
        'lambda_nip': lambda_nip,
        'lambda_dcn': lambda_dcn,
        'operations': operations,
        'dcn': dist_compression
    }
        
    dist = {'forensics_classes': forensics_classes}
    dist.update(distribution)

    return tf_ops, dist


# @coreutils.logCall
def train_manipulation_nip(tf_ops, training, distribution, data, directories=None, overwrite=False):
    """
    Jointly train the NIP and the FAN models. Training progress and TF checkpoints are saved periodically to the specified directories.
    
    Input parameters:
    
    All input parameters are dictionaries with the following keys:
    
    :param tf_ops: {
        sess     - TF session
        nip      - NIP instance
        fan      - FAN instance
        loss     - TF operation for the loss function
        opt      - TF operation for the optimization step
        lr       - TF placeholder for the learning rate
        lambda   - TF placeholder for the regularization strength
    }
    
    :param training: {
        camera_name           - name of the camera
        use_pretrained_nip    - boolean flag to enable/disable loading a pre-trained model
        lambda_nip            - regularization strength to control the trade-off between objectives (image quality)
        lambda_dcn            - regularization strength to control the trade-off between objectives (compression quality)
        run_number            - number of the current run ()
        n_epochs              - number of training epochs
        learning_rate         - value of the learning rate
    }
    
    :param distribution: {
      channel_jpeg_quality    - JPEG quality level in the distribution channel
      jpeg_approximation      - JPEG approximation mode
      forensics_classes       - names of classes for the FAN to distinguish
      channel_downsampling    - sub-sampling mode in the channel  
    }
            
    :param data: {
        training: {
            x: (N,h,w,4)      - input RAW images (patches will be sampled while training)
            y: (N,2h,2w,3)    - corresponding developed RGB images (patches will be sampled while training)
        }  
        validation: {
            x: (N,p,p,4)      - input RAW patches (for validation)
            y: (N,2p,2p,3)    - corresponding developed RGB patches (for validation)
        }
    }
    
    :param directories: {
        root             - the root output directory for storing training progress and model snapshots 
                           (default: './data/raw/train_manipulation/')
        nip_snapshots    - root directory with pre-trained NIP models 
                           (default: './data/raw/nip_model_snapshots/')
    }
    
    """
    
    # Apply default settings
    directories_def = {'root': './data/raw/train_manipulation/', 'nip_snapshots': './data/raw/nip_model_snapshots/'}
    if directories is not None: 
        directories_def.update(directories)    
    directories = directories_def
    
    # Check if all necessary keys are present
    if any([x not in tf_ops for x in ['sess', 'nip', 'fan', 'loss', 'opt', 'lr', 'lambda_nip', 'lambda_dcn']]):
        raise RuntimeError('Missing keys in the tf_ops dictionary! {}'.format(tf_ops.keys()))
        
    if any([x not in training for x in ['camera_name', 'use_pretrained_nip', 'lambda_nip', 'lambda_dcn', 'run_number', 'n_epochs', 'learning_rate']]):
        raise RuntimeError('Missing keys in the training dictionary! {}'.format(training.keys()))

    if any([x not in distribution for x in ['downsampling', 'compression', 'forensics_classes', 'compression_params']]):
        raise RuntimeError('Missing keys in the distribution dictionary! {}'.format(distribution.keys()))

    if data is None:
        raise ValueError('Training data seems not to be loaded!')

    try:
        if training['feed'] != 'rgb':
            batch_x, batch_y = data.next_training_batch(0, 5, training['patch_size'] * 2)
            if batch_x.shape != (5, training['patch_size'], training['patch_size'], 4) or batch_y.shape != (5, 2 * training['patch_size'], 2 * training['patch_size'], 3):
                raise ValueError('The training batch returned by the dataset is of invalid size!')

    except Exception as e:
        raise ValueError('Data set error: {}'.format(e))

    print('\n## Training NIP/FAN for manipulation detection: cam={} / lr={:.4f} / run={:3d} / epochs={}, root={}'.format(training['camera_name'], training['lambda_nip'], training['run_number'], training['n_epochs'], directories['root']), flush=True)

    # Construct output directory - some example paths:
    #  root / camera_name / *Net / fixed-nip / 001 /
    #  root / camera_name / *Net / lr-0.1000 / lc-0.1000 / 001 /
    #  root / camera_name / *Net / fixed-nip / lc-0.1000 / 001 /
    nip_save_dir = [directories['root'], training['camera_name'], tf_ops['nip'].class_name]
    if 'nip' in training['trainable']:
        nip_save_dir.append('ln-{:0.4f}'.format(training['lambda_nip']))
    else:
        nip_save_dir.append('fixed-nip')
    if 'dcn' in training['trainable']:
        nip_save_dir.append('lc-{:0.4f}'.format(training['lambda_dcn']))
    nip_save_dir.append('{:03d}'.format(training['run_number']))
    
    nip_save_dir = os.path.join(*nip_save_dir)
    print('(progress) ->', nip_save_dir)

    model_directory = os.path.join(nip_save_dir, 'models')
    print('(model) ---->', model_directory)

    if os.path.exists(nip_save_dir) and not overwrite:
        print('Directory exists, skipping...')
        return model_directory

    # Setup flags for trainable components
    joint_opt = ['fan']
    joint_opt.extend(sorted(training['trainable']))
    joint_opt = '+'.join(joint_opt)
    joint_optimization = ['nip' in training['trainable'], 'dcn' in training['trainable']]

    # Basic setup
    problem_description = 'manipulation detection'
    patch_size = training['patch_size']
    batch_size = training['batch_size']
    sampling_rate = training['sampling_rate']

    learning_rate_decay_schedule = 100
    learning_rate_decay_rate = 0.90

    # Setup the arrays for storing the current batch - randomly sampled from full-resolution images
    learning_rate = training['learning_rate']

    # Initialize models
    tf_ops['fan'].init()
    tf_ops['nip'].init()
    tf_ops['sess'].run(tf.global_variables_initializer())

    if training['use_pretrained_nip'] and training['feed'] == 'raw':
        tf_ops['nip'].load_model(os.path.join(directories['nip_snapshots'], training['camera_name'], tf_ops['nip'].scoped_name))
    
    if distribution['compression'] == 'dcn':
        tf_ops['dcn'].load_model(distribution['compression_params']['dirname'])

    n_batches = data.count_training // batch_size

    model_list = ['nip', 'fan']

    # Containers for storing loss progression
    loss_epoch = {key: deque(maxlen=n_batches) for key in model_list}
    loss_epoch['similarity-loss'] = deque(maxlen=n_batches)
    loss_last_k_epochs = {key: deque(maxlen=10) for key in model_list}
    loss_last_k_epochs['similarity-loss'] = deque(maxlen=n_batches)
    
    # Create a function which generates labels for each batch
    def batch_labels(batch_size, n_classes):
        return np.concatenate([x * np.ones((batch_size,), dtype=np.int32) for x in range(n_classes)])

    n_classes = len(distribution['forensics_classes'])
    batch_l = batch_labels(batch_size, n_classes)

    # Collect memory usage (seems to be leaking in matplotlib)
    collect_memory_stats = {'tf': False, 'ram': True}
    memory = {'tf-ram': [], 'tf-vars': [], 'cpu-proc': [], 'cpu-resource': [] }

    # Collect and print training summary
    training_summary = OrderedDict()
    training_summary['Problem'] = '{}'.format(problem_description)
    training_summary['Classes'] = '{}'.format(distribution['forensics_classes'])
    training_summary['Channel Downsampling'] = '{}'.format(distribution['downsampling'])
    training_summary['Channel Compression'] = '{}'.format(tf_ops['dcn'].summary() if tf_ops['dcn'] is not None else None)
    training_summary['Channel Compression Parameters'] = '{}'.format(distribution['compression_params'])
    training_summary['Camera name'] = '{}'.format(training['camera_name'])
    training_summary['Joint optimization'] = '{}'.format(joint_opt)
    training_summary['NIP Regularization'] = '{}'.format(training['lambda_nip'])
    training_summary['DCN Regularization'] = '{}'.format(training['lambda_dcn'])
    training_summary['FAN model'] = '{}'.format(tf_ops['fan'].summary())
    training_summary['NIP model'] = '{}'.format(tf_ops['nip'].summary())
    training_summary['NIP loss'] = '{}'.format(tf_ops['nip'].loss_metric)
    training_summary['Use pre-trained NIP'] = '{}'.format(training['use_pretrained_nip'])
    training_summary['# Epochs'] = '{}'.format(training['n_epochs'])
    training_summary['Patch size'] = '{}'.format(patch_size)
    training_summary['Batch size'] = '{}'.format(batch_size)
    training_summary['Learning rate'] = '{}'.format(training['learning_rate'])
    training_summary['Learning rate decay schedule'] = '{}'.format(learning_rate_decay_schedule)
    training_summary['Learning rate decay rate'] = '{}'.format(learning_rate_decay_rate)
    if training['feed'] != 'rgb':
        training_summary['# train. images'] = '{}'.format(data['training']['x'].shape)
        training_summary['# valid. images'] = '{}'.format(data['validation']['x'].shape)
        training_summary['# batches'] = '{}'.format(batch_x.shape)
    training_summary['NIP input patch'] = '{}'.format(tf_ops['nip'].x.shape)
    training_summary['NIP output patch'] = '{}'.format(tf_ops['nip'].y.shape)
    training_summary['FAN input patch'] = '{}'.format(tf_ops['fan'].x.shape)
    if any(collect_memory_stats.values()):
        training_summary['memory_consumption'] = memory

    print('')
    for k, v in training_summary.items():
        print('{:30s}: {}'.format(k, v))
    print('\n', flush=True)

    with tqdm.tqdm(total=training['n_epochs'], ncols=120, desc='Train') as pbar:
        
        epoch = 0
        conf = np.identity(len(distribution['forensics_classes']))

        for epoch in range(0, training['n_epochs']):

            for batch_id in range(n_batches):

                # Extract random patches for the current batch of images
                if training['feed'] == 'raw':
                    batch_x, batch_y = data.next_training_batch(batch_id, batch_size, 2 * patch_size)
                else:
                    batch_x = data.next_training_batch(batch_id, batch_size, 2 * patch_size)
                    batch_y = batch_x

                if any(joint_optimization):
                    # Make custom optimization step                    
                    comb_loss, nip_loss, _ = tf_ops['sess'].run([tf_ops['loss'], tf_ops['nip'].loss, tf_ops['opt']], feed_dict={
                        tf_ops['nip'].x if training['feed'] == 'raw' else tf_ops['nip'].yy: batch_x,
                        tf_ops['nip'].y_gt: batch_y,
                        tf_ops['fan'].y: batch_l,
                        tf_ops['lr']: learning_rate,
                        tf_ops['lambda_nip']: training['lambda_nip'],
                        tf_ops['lambda_dcn']: training['lambda_dcn']
                    })                    
                    
                    loss_epoch['nip'].append(nip_loss)
                else:
                    # Update only the forensics network
                    comb_loss = tf_ops['fan'].training_step(batch_x, batch_l, learning_rate)
                    nip_loss = np.nan

                loss_epoch['fan'].append(comb_loss)
                loss_epoch['nip'].append(nip_loss)

            # Average and record loss values
            for model_name in model_list:
                tf_ops[model_name].performance['loss']['training'].append(float(np.mean(loss_epoch[model_name])))
                loss_last_k_epochs[model_name].append(tf_ops[model_name].performance['loss']['training'][-1])

            if epoch % sampling_rate == 0:

                # Validate the NIP model
                if joint_optimization[0]:
                    values = validation.validate_nip(tf_ops['nip'], data, nip_save_dir, epoch=epoch, show_ref=True, loss_type=tf_ops['nip'].loss_metric)
                    for metric, val_array in zip(['ssim', 'psnr', 'loss'], values):
                        tf_ops['nip'].performance[metric]['validation'].append(float(np.mean(val_array)))
                        
                # Validate the DCN model
                if joint_optimization[1]:
                    values = validation.validate_dcn(tf_ops['dcn'], data, nip_save_dir, epoch=epoch, show_ref=True)
                    for metric, val_array in zip(['ssim', 'psnr', 'loss', 'entropy'], values):
                        tf_ops['dcn'].performance[metric]['validation'].append(float(np.mean(val_array)))                    

                # Validate the forensics network
                accuracy = validation.validate_fan(tf_ops['fan'], data, lambda x: batch_labels(x, n_classes), n_classes)
                tf_ops['fan'].performance['accuracy']['validation'].append(accuracy)

                # Confusion matrix
                conf = validation.confusion(tf_ops['fan'], data, lambda x: batch_labels(x, n_classes))

                # Visualize current progress
                # TODO Memory is leaking here - looks like some problem in matplotlib - skip for now
                # validation.visualize_manipulation_training(model, fan, conf, epoch, nip_save_dir, classes=distribution['forensics_classes'])

                # Save progress stats
                validation.save_training_progress(training_summary, tf_ops['nip'], tf_ops['fan'], tf_ops['dcn'], conf, nip_save_dir)

                # Save models
                tf_ops['fan'].save_model(os.path.join(model_directory, tf_ops['fan'].scoped_name), epoch)

                if joint_optimization[0]:
                    tf_ops['nip'].save_model(os.path.join(model_directory, tf_ops['nip'].scoped_name), epoch)

                if isinstance(tf_ops['dcn'], compression.DCN) and joint_optimization[1]:
                    tf_ops['dcn'].save_model(os.path.join(model_directory, tf_ops['dcn'].scoped_name), epoch)

                # Monitor memory usage
                # gc.collect()
                if collect_memory_stats['tf']:
                    memory['tf-ram'].append(round(tf_helpers.memory_usage_tf(tf_ops['sess']) / 1024 / 1024, 1))
                    memory['tf-vars'].append(round(tf_helpers.memory_usage_tf_variables() / 1024 / 1024, 1))

                if collect_memory_stats['ram']:
                    memory['cpu-proc'].append(round(coreutils.memory_usage_proc(), 1))
                    memory['cpu-resource'].append(round(coreutils.memory_usage_resource(), 1))

            if epoch % learning_rate_decay_schedule == 0:
                learning_rate *= learning_rate_decay_rate

            # Update the progress bar
            progress_stats = {
                'nip': np.log10(np.mean(loss_last_k_epochs['nip'])).round(1),                
                'fan': np.mean(loss_last_k_epochs['fan']),
                'acc': tf_ops['fan'].performance['accuracy']['validation'][-1],
            }
            
            if distribution['compression'] == 'dcn' and joint_optimization[1]:
                progress_stats['dcn'] = tf_ops['dcn'].performance['ssim']['validation'][-1]
                progress_stats['H'] = tf_ops['dcn'].performance['entropy']['validation'][-1]

            if len(tf_ops['nip'].performance['psnr']['validation']) > 0:
                progress_stats['psnr'] = tf_ops['nip'].performance['psnr']['validation'][-1]

            if collect_memory_stats['ram']:
                progress_stats['ram'] = round(memory['cpu-proc'][-1]//1024, 2)

            pbar.set_postfix(**progress_stats)
            pbar.update(1)

    # Plot final results
    values = validation.validate_nip(tf_ops['nip'], data, nip_save_dir, epoch=epoch, show_ref=True, loss_type='L2')
    for metric, val_array in zip(['ssim', 'psnr', 'loss'], values):
        tf_ops['nip'].performance[metric]['validation'].append(float(np.mean(val_array)))

    if isinstance(tf_ops['dcn'], compression.DCN):
        values = validation.validate_dcn(tf_ops['dcn'], data, nip_save_dir, epoch=epoch, show_ref=True)
        for metric, val_array in zip(['ssim', 'psnr', 'loss', 'entropy'], values):
            tf_ops['dcn'].performance[metric]['validation'].append(float(np.mean(val_array)))
        
    # Compute confusion matrix
    conf = validation.confusion(tf_ops['fan'], data, lambda x: batch_labels(x, n_classes))

    # Save model progress
    validation.save_training_progress(training_summary, tf_ops['nip'], tf_ops['fan'], tf_ops['dcn'], conf, nip_save_dir)

    # Visualize current progress
    validation.visualize_manipulation_training(tf_ops['nip'], tf_ops['fan'], tf_ops['dcn'], conf, epoch, nip_save_dir, classes=distribution['forensics_classes'])

    # Save models
    print('Saving models...', end='')

    tf_ops['fan'].save_model(os.path.join(model_directory, tf_ops['fan'].scoped_name), epoch)
    print(' fan', end='')
    
    if joint_optimization[0]:
        tf_ops['nip'].save_model(os.path.join(model_directory, tf_ops['nip'].scoped_name), epoch)
        print(' nip', end='')
    
    if isinstance(tf_ops['dcn'], compression.DCN) and joint_optimization[1]:
        tf_ops['dcn'].save_model(os.path.join(model_directory, tf_ops['dcn'].scoped_name), epoch)
        print(' dcn', end='')
    
    print('')  # \newline
    
    return model_directory
