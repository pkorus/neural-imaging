#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import logging
import argparse

import numpy as np
import scipy.fftpack as sfft

from helpers import coreutils

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

supported_pipelines = ['UNet', 'DNet', 'INet']


def fft_log_norm(x):
    y = np.zeros_like(x)
    for i in range(3):
        y[:, :, i] = np.abs(sfft.fft2(x[:, :, i]))
        y[:, :, i] = sfft.fftshift(y[:, :, i])
        y[:, :, i] = np.log(10 + y[:, :, i])
        y[:, :, i] = nm(y[:, :, i])
    return y


def nm(x):
    return (x - x.min()) / (x.max() - x.min())


def quick_show(ax, x, l):
    ax.imshow(x)
    ax.set_title(l)
    ax.set_xticks([])
    ax.set_yticks([])


def compare_nips(camera, pipeline, model_a_dirname, model_b_dirname, ps=128, image_id=None, root_dir='./data/raw', output_dir=None):
    """
    Display a comparison of two variants of a neural imaging pipeline.
    :param camera: camera name (e.g., 'Nikon D90')
    :param pipeline: neural pipeline name (e.g., 'UNet')
    :param model_a_dirname: directory with the first variant of the model
    :param model_b_dirname: directory with the second variant of the model
    :param ps: patch size (patch will be taken from the middle)
    :param image_id: index of the test image
    :param root_dir: root data directory
    :param output_dir: set an output directory if the figure should be saved (matplotlib2tikz will be used)
    """

    supported_cameras = coreutils.listdir(os.path.join(root_dir,  'nip_model_snapshots'), '.*')

    if ps < 4 or ps > 2048:
        raise ValueError('Patch size seems to be invalid!')

    if pipeline not in supported_pipelines:
        raise ValueError('Unsupported pipeline model ({})! Available models: {}'.format(pipeline, ', '.join(supported_pipelines)))

    if camera not in supported_cameras:
        raise ValueError('Camera data not found ({})! Available cameras: {}'.format(camera, ', '.join(supported_cameras)))

    image_id = image_id or 0

    # Lazy imports to minimize delay for invalid command line parameters
    import re
    import imageio as io
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from models import pipelines
    from skimage.measure import compare_psnr, compare_ssim

    # Find available Bayer stacks for the camera
    dirname = os.path.join(root_dir, 'nip_training_data', camera)
    files = coreutils.listdir(dirname, '.*\.npy')

    if len(files) == 0:
        print('ERROR Not training files found for the given camera model!')
        sys.exit(2)

    # Get model class instance
    nip_model = getattr(pipelines, pipeline)

    # Construct the NIP models
    g1 = tf.Graph()
    g2 = tf.Graph()
    sess1 = tf.Session(graph=g1)
    sess2 = tf.Session(graph=g2)

    model_a = nip_model(sess1, g1)
    model_a.init()
    model_a.load_model(camera, model_a_dirname)

    model_b = nip_model(sess2, g2)
    model_b.init()
    model_b.load_model(camera, os.path.join(model_b_dirname, '{nip-model}'))

    log.info('Model A: {}'.format(model_a.summary()))
    log.info('Model B: {}'.format(model_b.summary()))

    # Load sample data
    sample_x = np.load(os.path.join(dirname, files[image_id]))
    sample_x = np.expand_dims(sample_x, axis=0)
    xx = (sample_x.shape[2] - ps) // 2
    yy = (sample_x.shape[1] - ps) // 2
    log.info('Using image {}'.format(files[image_id]))
    log.info('Cropping patch from input image x={}, y={}, size={}'.format(xx, yy, ps))
    sample_x = sample_x[:, yy:yy+ps, xx:xx+ps, :].astype(np.float32) / (2**16 - 1)

    # Develop images
    sample_ya = model_a.process(sample_x)
    sample_yb = model_b.process(sample_x)
    
    target_y = io.imread(os.path.join(dirname, files[image_id].replace('.npy', '.png')))
    target_y = target_y[2*yy:2*(yy+ps), 2*xx:2*(xx+ps), :].astype(np.float32) / (2**8 - 1)

    # Plot images
    fig = plt.figure(figsize=(9, 9))

    quick_show(fig.add_subplot(3, 3, 1), target_y, '(T)arget')
    quick_show(fig.add_subplot(3, 3, 2), sample_ya.squeeze(), '{}(A) {:.1f} dB / {:.3f}'.format(pipeline, compare_psnr(target_y, sample_ya.squeeze(), data_range=1.0), compare_ssim(target_y, sample_ya.squeeze(), multichannel=True)))
    quick_show(fig.add_subplot(3, 3, 4), sample_yb.squeeze(), '{}(B) {:.1f} dB / {:.3f}'.format(pipeline, compare_psnr(target_y, sample_yb.squeeze(), data_range=1.0), compare_ssim(target_y, sample_yb.squeeze(), multichannel=True)))

    # Compute and plot difference images
    diff_a = np.abs(sample_ya.squeeze() - target_y)
    diff_a_mean = diff_a.mean()
    diff_a = nm(diff_a)

    diff_b = np.abs(sample_yb.squeeze() - target_y)
    diff_b_mean = diff_b.mean()
    diff_b = nm(diff_b)

    diff_ab = np.abs(sample_yb.squeeze() - sample_ya.squeeze())
    diff_ab_mean = diff_ab.mean()
    diff_ab = nm(diff_ab)

    quick_show(fig.add_subplot(3, 3, 3), diff_a, 'T - A: mean abs {:.3f}'.format(diff_a_mean))
    quick_show(fig.add_subplot(3, 3, 7), diff_b, 'T - B: mean abs {:.3f}'.format(diff_b_mean))
    quick_show(fig.add_subplot(3, 3, 5), diff_ab, 'A - B: mean abs {:.3f}'.format(diff_ab_mean))

    # Compute and plot spectra
    fft_a = fft_log_norm(diff_a)
    fft_b = fft_log_norm(diff_b)
    # fft_ab = nm(np.abs(fft_a - fft_b))
    fft_ab = nm(np.abs(fft_log_norm(sample_yb.squeeze()) - fft_log_norm(sample_ya.squeeze())))
    
    quick_show(fig.add_subplot(3, 3, 6), fft_a, 'FFT(T - A)')
    quick_show(fig.add_subplot(3, 3, 8), fft_b, 'FFT(T - B)')
    quick_show(fig.add_subplot(3, 3, 9), fft_ab, 'FFT(A) - FFT(B)')

    if output_dir is not None:
        from matplotlib2tikz import save as tikz_save
        dcomp = [x for x in model_b_dirname.split('/') if re.match('(lr-.*|[0-9]{3})', x)]
        tikz_save('{}/examples-{}-{}-{}-{}-{}.tex'.format(output_dir, camera, pipeline, image_id, dcomp[0], dcomp[1]),
                  figureheight='8cm', figurewidth='8cm', strict=False)
    else:
        fig.tight_layout()
        plt.show(fig)

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--nip', dest='nip', action='store', help='imaging pipeline ({})'.format(supported_pipelines))
    parser.add_argument('--image', dest='image', action='store', default=0, type=int,
                        help='image id (n-th image in the camera\'s directory')
    parser.add_argument('--patch', dest='patch', action='store', default=128, type=int,
                        help='patch size')
    parser.add_argument('--a', dest='model_a_dir', action='store', default='./data/raw/nip_model_snapshots',
                        help='path to first model (TF checkpoint dir)')
    parser.add_argument('--b', dest='model_b_dir', action='store', default='./data/raw/nip_model_snapshots',
                        help='path to second model (TF checkpoint dir)')
    parser.add_argument('--dir', dest='dir', action='store', default='./data/raw/',
                        help='root directory with images and training data')
    parser.add_argument('--out', dest='out', action='store', default=None,
                        help='output directory for TikZ output (if set, the figure is not displayed)')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    try:
        compare_nips(args.camera, args.nip, args.model_a_dir, args.model_b_dir, args.patch, args.image, args.dir, args.out)
    except Exception as error:
        log.error('Error while running the comparison function: {}'.format(error))


if __name__ == "__main__":
    main()

