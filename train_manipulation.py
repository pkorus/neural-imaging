#!/usr/bin/env python
# coding: utf-8

# Basic imports
import gc
import os
import argparse
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

# Helper functions
from helpers import coreutils, tf_helpers, validation, loading


@coreutils.logCall
def construct_models(nip_model, patch_size=128, distribution_jpeg=50, distribution_down='pool', loss_metric='L2'):
    """
    Setup the TF model of the entire acquisition and distribution workflow.
    :param nip_model: name of the NIP class
    :param patch_size: patch size for manipulation training (raw patch - rgb patches will be 4 times as big)
    :param distribution_jpeg: JPEG quality level in the distribution channel
    :param distribution_down: Sub-sampling method in the distribution channel ('pool' or 'bilin')
    :param loss_metric: NIP loss metric: L2, L1 or SSIM
    """
    # Sanitize inputs
    if patch_size < 32 or patch_size > 512:
        raise ValueError('The patch size ({}) looks incorrect, typical values should be >= 32 and <= 512'.format(patch_size))

    if distribution_jpeg < 1 or distribution_jpeg > 100:
        raise ValueError('Invalid JPEG quality level ({})'.format(distribution_jpeg))

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

        # Sharpen    
        im_shr = tf_helpers.tf_sharpen(model.y, 0, hsv=True)

        # Bilinear resampling
        im_res = tf.image.resize_images(model.y, [tf.shape(model.y)[1] // 2, tf.shape(model.y)[1] // 2])
        im_res = tf.image.resize_images(im_res, [tf.shape(model.y)[1], tf.shape(model.y)[1]])

        # Gaussian filter
        im_gauss = tf_helpers.tf_gaussian(model.y, 5, 4)

        # Mild JPEG
        tf_jpg = DJPG(sess, tf.get_default_graph(), model.y, None, quality=80, rounding_approximation='sin')
        im_jpg = tf_jpg.y

        # Setup operations for detection
        operations = (model.y, im_shr, im_gauss, im_jpg, im_res)
        forensics_classes = ['native', 'sharpen', 'gaussian', 'jpg', 'resample']

        n_classes = len(operations)

        # Concatenate outputs from multiple post-processing paths ------------------------------------------------------
        y_concat = tf.concat(operations, axis=0)

        # Add sub-sampling and JPEG compression in the channel ---------------------------------------------------------
        if distribution_down == 'pool':
            imb_down = tf.nn.avg_pool(y_concat, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME', name='post_downsample')
        elif distribution_down == 'bilin':
            imb_down = tf.image.resize_images(y_concat, [tf.shape(y_concat)[1] // 2, tf.shape(y_concat)[1] // 2])
        else:
            raise ValueError('Unsupported channel down-sampling {}'.format(distribution_down))
            
        jpg = DJPG(sess, tf.get_default_graph(), imb_down, model.x, quality=distribution_jpeg, rounding_approximation='sin')
        imb_out = jpg.y

    # Add manipulation detection
    fan = FAN(sess, tf.get_default_graph(), n_classes=n_classes, x=imb_out, nip_input=model.x, n_convolutions=4)
    print('Forensics network parameters: {:,}'.format(fan.count_parameters()))

    # Setup a combined loss and training op
    with tf.name_scope('combined_optimization') as scope:
        nip_fw = tf.placeholder(tf.float32, name='nip_weight')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        loss = fan.loss + nip_fw * model.loss
        adam = tf.train.AdamOptimizer(learning_rate=lr, name='adam')
        opt = adam.minimize(loss, name='opt_combined')
    
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    return sess, model, fan, {'loss': loss, 'opt': opt, 'lr': lr, 'lambda': nip_fw}, operations, {'channel_jpeg_quality': distribution_jpeg, 'forensics_classes': forensics_classes, 'downsampling': distribution_down}


# @coreutils.logCall
def train_manipulation_nip(camera_name, nip_weight, run_number, n_epochs=1001, learning_rate=1e-4,
                           use_pretrained_nip=True, root_directory='./data/raw/train_manipulation/',
                           nip_root_directory='./data/raw/nip_model_snapshots/'):
    """
    Jointly train the NIP and the FAN models.
    :param camera_name: name of the camera model (e.g., "Nikon D90")
    :param nip_weight: NIP regularization strength (set 0 to disable joint optimization, and train the FAN only)
    :param run_number: number for labeling evaluation runs (used in output directory name)
    :param n_epochs: number of training epochs
    :param learning_rate: learning rate
    :param use_pretrained_nip: whether the NIP model should be initialized with a pre-trained network
    :param root_directory: the root output directory for storing training progress and model snapshots
    :param nip_root_directory: root directory with pre-trained NIP models
    """

    print('\n## Training NIP/FAN for manipulation detection: cam={} / lr={:.4f} / run={:3d} / epochs={}, root={}'.format(camera_name, nip_weight, run_number, n_epochs, root_directory), flush=True)

    nip_save_dir = os.path.join(root_directory, camera_name, '{nip-model}', 'lr-{:0.4f}'.format(nip_weight), '{:03d}'.format(run_number))
    print('(progress) ->', nip_save_dir)

    model_directory = os.path.join(nip_save_dir.replace('{nip-model}', type(model).__name__), 'models')
    print('(model) ---->', model_directory)

    # Enable joint optimization if NIP weight is non-zero
    joint_optimization = nip_weight != 0

    # Basic setup
    problem_description = 'manipulation detection'
    patch_size = 128
    batch_size = 20
    sampling_rate = 50

    learning_rate_decay_schedule = 100
    learning_rate_decay_rate = 0.85

    # Setup the arrays for storing the current batch - randomly sampled from full-resolution images
    H, W = data_x.shape[1:3]

    batch_x = np.zeros((batch_size, patch_size, patch_size, 4), dtype=np.float32)
    batch_y = np.zeros((batch_size, 2 * patch_size, 2 * patch_size, 3), dtype=np.float32)

    # Initialize models
    fan.init()
    model.init()
    sess.run(tf.global_variables_initializer())

    if use_pretrained_nip:
        model.load_model(camera_name=camera_name, out_directory_root=nip_root_directory)

    n_batches = data_x.shape[0] // batch_size

    model_list = ['nip', 'fan']
    model_obj = [model, fan]

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
    memory = {'tf-ram': [], 'tf-vars': [], 'cpu-proc': [], 'cpu-resource': [] }

    # Collect and print training summary
    training_summary = OrderedDict()
    training_summary['Problem'] = '{}'.format(problem_description)
    training_summary['Classes'] = '{}'.format(distribution['forensics_classes'])
    training_summary['Channel sub-sampling'] = '{}'.format(distribution['downsampling'])
    training_summary['Channel JPEG'] = '{}'.format(distribution['channel_jpeg_quality'])
    training_summary['Camera name'] = '{}'.format(camera_name)
    training_summary['Joint optimization'] = '{}'.format(joint_optimization)
    training_summary['NIP Regularization'] = '{}'.format(nip_weight)
    training_summary['FAN model'] = '{}'.format(fan.summary())
    training_summary['NIP model'] = '{}'.format(model.summary())
    training_summary['NIP loss'] = '{}'.format(model.loss_metric)
    training_summary['Use pre-trained NIP'] = '{}'.format(use_pretrained_nip)
    training_summary['# Epochs'] = '{}'.format(n_epochs)
    training_summary['Patch size'] = '{}'.format(patch_size)
    training_summary['Batch size'] = '{}'.format(batch_size)
    training_summary['Learning rate'] = '{}'.format(learning_rate)
    training_summary['Learning rate decay schedule'] = '{}'.format(learning_rate_decay_schedule)
    training_summary['Learning rate decay rate'] = '{}'.format(learning_rate_decay_rate)
    training_summary['# train. images'] = '{}'.format(data_x.shape)
    training_summary['# valid. images'] = '{}'.format(valid_x.shape)
    training_summary['# batches'] = '{}'.format(batch_x.shape)
    training_summary['NIP input patch'] = '{}'.format(model.x.shape)
    training_summary['NIP output patch'] = '{}'.format(model.y.shape)
    training_summary['FAN input patch'] = '{}'.format(fan.x.shape)
    training_summary['memory_consumption'] = memory

    print('\n')
    for k, v in training_summary.items():
        print('{:30s}: {}'.format(k, v))
    print('\n', flush=True)

    with tqdm.tqdm(total=n_epochs, ncols=120, desc='Train') as pbar:
        
        epoch = 0
        conf = np.identity(len(distribution['forensics_classes']))

        for epoch in range(0, n_epochs):

            # Fill the batch with random crops of the images
            for batch_id in range(n_batches):

                # Extract random patches for the current batch of images
                for b in range(batch_size):
                    xx = np.random.randint(0, W - patch_size)
                    yy = np.random.randint(0, H - patch_size)
                    batch_x[b, :, :, :] = data_x[batch_id * batch_size + b, yy:yy + patch_size, xx:xx + patch_size, :].astype(np.float) / (2**16 - 1)
                    batch_y[b, :, :, :] = data_y[batch_id * batch_size + b, (2*yy):(2*yy + 2*patch_size), (2*xx):(2*xx + 2*patch_size), :].astype(np.float) / (2**8 - 1)

                if joint_optimization:
                    # Make custom optimization step                    
                    comb_loss, nip_loss, _ = sess.run([train_setup['loss'], model.loss, train_setup['opt']], feed_dict={
                        model.x: batch_x,
                        model.y_gt: batch_y,
                        fan.y: batch_l,
                        train_setup['lr']: learning_rate,
                        train_setup['lambda']: nip_weight                        
                    })                    
                    
                    loss_epoch['nip'].append(nip_loss)
                else:
                    # Update only the forensics network
                    comb_loss = fan.training_step(batch_x, batch_l, learning_rate)
                    nip_loss = np.nan

                loss_epoch['fan'].append(comb_loss)
                loss_epoch['nip'].append(nip_loss)

            # Average and record loss values
            for model_name, mod in zip(model_list, model_obj):
                mod.train_perf['loss'].append(float(np.mean(loss_epoch[model_name])))
                loss_last_k_epochs[model_name].append(mod.train_perf['loss'][-1])

            if epoch % sampling_rate == 0:

                # Validate the NIP model
                if joint_optimization:
                    ssims, psnrs, v_losses = validation.validate_nip(model, valid_x[::50], valid_y[::50], None, epoch=epoch, show_ref=True, loss_type='L2')
                    model.valid_perf['ssim'].append(float(np.mean(ssims)))
                    model.valid_perf['psnr'].append(float(np.mean(psnrs)))
                    model.valid_perf['loss'].append(float(np.mean(v_losses)))

                # Validate the forensics network
                accuracy = validation.validate_fan(fan, valid_x[::1], lambda x: batch_labels(x, n_classes), n_classes)
                fan.valid_perf['accuracy'].append(accuracy)

                # Confusion matrix
                conf = validation.confusion(fan, valid_x, lambda x: batch_labels(x, n_classes))

                # Visualize current progress
                # TODO Memory is leaking here - looks like some problem in matplotlib - skip for now
                # validation.visualize_manipulation_training(model, fan, conf, epoch, nip_save_dir, classes=distribution['forensics_classes'])

                # Save progress stats
                validation.save_training_progress(training_summary, model, fan, conf, nip_save_dir)

                # Monitor memory usage
                # gc.collect()
                memory['tf-ram'].append(round(tf_helpers.memory_usage_tf(sess) / 1024 / 1024, 1))
                memory['tf-vars'].append(round(tf_helpers.memory_usage_tf_variables() / 1024 / 1024, 1))
                memory['cpu-proc'].append(round(coreutils.memory_usage_proc(), 1))
                memory['cpu-resource'].append(round(coreutils.memory_usage_resource(), 1))

            if epoch % learning_rate_decay_schedule == 0:
                learning_rate = learning_rate * learning_rate_decay_rate

            pbar.set_postfix(nip=np.log10(np.mean(loss_last_k_epochs['nip'])).round(1),
                             fan=np.mean(loss_last_k_epochs['fan']),
                             acc=fan.valid_perf['accuracy'][-1],
                             psnr=model.valid_perf['psnr'][-1] if len(model.valid_perf['psnr']) > 0 else np.nan,
                             ram=round(memory['cpu-proc'][-1]//1024, 2))
            pbar.update(1)

    # Plot final results
    ssims, psnrs, v_losses = validation.validate_nip(model, valid_x[::50], valid_y[::50], nip_save_dir, epoch=epoch, show_ref=True, loss_type='L2')
    model.valid_perf['ssim'].append(float(np.mean(ssims)))
    model.valid_perf['psnr'].append(float(np.mean(psnrs)))
    model.valid_perf['loss'].append(float(np.mean(v_losses)))

    # Save model progress
    validation.save_training_progress(training_summary, model, fan, conf, nip_save_dir)

    # Visualize current progress
    validation.visualize_manipulation_training(model, fan, conf, epoch, nip_save_dir, classes=distribution['forensics_classes'])

    # Save models
    # Root     : train_manipulation / camera_name / {INet} / lr-01 / 001 / models / {INet/FAN}
    print('Saving models...')
    model.save_model(camera_name, os.path.join(model_directory, '{nip-model}'), epoch)
    fan.save_model(os.path.join(model_directory, 'FAN'), epoch)


@coreutils.logCall
def batch_training(nip_model, camera_names=None, root_directory=None, loss_metric='L2', jpeg_q=50, use_pretrained=True, end_repetition=10, start_repetition=0, n_epochs=1001):
    """
    Repeat training for multiple NIP regularization strengths.
    """
    global sess, model, fan, train_setup, distribution, data_x, data_y, valid_x, valid_y

    # Training set setup
    valid_patch_size = 128
    n_patches = 100

    # Experiment setup
    camera_names = camera_names or ['Nikon D90', 'Nikon D7000', 'Canon EOS 5D', 'Canon EOS 40D']
    regularization_strengths = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1]

    # Construct the TF model
    sess, model, fan, train_setup, _, distribution = construct_models(nip_model, distribution_jpeg=jpeg_q, loss_metric=loss_metric)

    for camera_name in camera_names:
        
        print('\n# Loading data for camera {}'.format(camera_name))
        
        # Load the dataset for the given camera
        data_directory = os.path.join('./data/raw/nip_training_data/', camera_name)

        # Find available images
        files, val_files = loading.discover_files(data_directory)

        # Load training / validation data
        data_x, data_y = loading.load_fullres(files, data_directory)
        valid_x, valid_y = loading.load_patches(val_files, data_directory, valid_patch_size, n_patches, discard_flat=True)

        # Repeat evaluation
        for rep in range(start_repetition, end_repetition):
            for reg in regularization_strengths:
                train_manipulation_nip(camera_name, reg, rep, n_epochs=n_epochs, root_directory=root_directory, use_pretrained_nip=use_pretrained)


def main():
    parser = argparse.ArgumentParser(description='NIP & FAN optimization for manipulation detection')
    parser.add_argument('--nip', dest='nip_model', action='store',
                        help='the NIP model (INet, UNet, DNet)')
    parser.add_argument('--jpeg', dest='jpeg', action='store', default=50, type=int,
                        help='JPEG quality level in the distribution channel')
    parser.add_argument('--dir', dest='root_dir', action='store', default='./data/raw/train_manipulation_box/',
                        help='the root directory for storing results')
    parser.add_argument('--cam', dest='cameras', action='append',
                        help='add cameras for evaluation (repeat if needed)')
    parser.add_argument('--loss', dest='loss_metric', action='store', default='L2',
                        help='loss metric for the NIP (L2, L1, SSIM)')
    parser.add_argument('--scratch', dest='from_scratch', action='store_true', default=False,
                        help='train NIP from scratch (ignore pre-trained model)')
    parser.add_argument('--start', dest='start', action='store', default=0, type=int,
                        help='first iteration (default 0)')
    parser.add_argument('--end', dest='end', action='store', default=10, type=int,
                        help='last iteration (exclusive, default 10)')
    parser.add_argument('--epochs', dest='epochs', action='store', default=1001, type=int,
                        help='number of epochs (default 1001)')

    args = parser.parse_args()

    batch_training(args.nip_model, args.cameras, args.root_dir, args.loss_metric, args.jpeg, not args.from_scratch,
                   start_repetition=args.start, end_repetition=args.end, n_epochs=args.epochs)


if __name__ == "__main__":
    main()
