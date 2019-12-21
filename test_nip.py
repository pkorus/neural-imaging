#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import logging
import argparse

from helpers import coreutils

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

supported_pipelines = ['UNet', 'DNet', 'INet']


def develop_image(camera, pipeline, ps=128, image_id=None, root_dir='./data'):
    """
    Display a patch developed by a neural imaging pipeline.
    """

    supported_cameras = coreutils.listdir(os.path.join(root_dir,  'models', 'nip'), '.*')

    if ps < 4 or ps > 2048:
        raise ValueError('Patch size seems to be invalid!')

    if pipeline not in supported_pipelines:
        raise ValueError('Unsupported pipeline model ({})! Available models: {}'.format(pipeline, ', '.join(supported_pipelines)))

    if camera not in supported_cameras:
        raise ValueError('Camera data not found ({})! Available cameras: {}'.format(camera, ', '.join(supported_cameras)))

    image_id = image_id or 0

    # Lazy imports to minimize delay for invalid command line parameters
    import numpy as np
    import imageio as io
    import matplotlib.pylab as plt
    import tensorflow as tf
    from models import pipelines
    from skimage.measure import compare_psnr

    root_dirname = os.path.join(root_dir, 'models', 'nip')
    data_dirname = os.path.join(root_dir, 'raw', 'training_data', camera)
    files = coreutils.listdir(data_dirname, '.*\.npy')

    if len(files) == 0:
        print('ERROR Not training files found for the given camera model!')
        sys.exit(3)

    # Get model class instance
    nip_model = getattr(pipelines, pipeline)

    # Construct the NIP model
    tf.reset_default_graph()
    sess = tf.Session()

    model = nip_model(sess, tf.get_default_graph())
    log.info('Using NIP: {}'.format(model.summary()))
    log.info('Loading weights from: {}'.format(os.path.join(root_dirname, camera)))
    model.load_model(os.path.join(root_dirname, camera))

    # Load sample data
    sample_x = np.load(os.path.join(data_dirname, files[image_id]))
    sample_x = np.expand_dims(sample_x, axis=0)
    xx = (sample_x.shape[2] - ps) // 2
    yy = (sample_x.shape[1] - ps) // 2
    log.info('Using image {}'.format(files[image_id]))
    log.info('Cropping patch from input image x={}, y={}, size={}'.format(xx, yy, ps))
    sample_x = sample_x[:, yy:yy+ps, xx:xx+ps, :].astype(np.float32) / (2**16 - 1)
    sample_x = np.repeat(sample_x, 20, axis=0)

    sample_y = model.process(sample_x)
    sample_y = sample_y[0:1]
    target_y = io.imread(os.path.join(data_dirname, files[image_id].replace('.npy', '.png')))
    target_y = target_y[2*yy:2*(yy+ps), 2*xx:2*(xx+ps), :].astype(np.float32) / (2**8 - 1)
    psnr_value = compare_psnr(target_y, sample_y.squeeze(), data_range=1.0)
    log.info('PSNR={:.1f} dB'.format(psnr_value))

    # Plot the images
    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(sample_y.squeeze())
    plt.title('{}, PSNR={:.1f} dB'.format(type(model).__name__, psnr_value))
    plt.subplot(1, 2, 2)
    plt.imshow(target_y)
    plt.title('Target')
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--nip', dest='nip', action='store', help='imaging pipeline ({})'.format(supported_pipelines))
    parser.add_argument('--image', dest='image', action='store', default=0, type=int,
                        help='image id (n-th image in the camera\'s directory')
    parser.add_argument('--patch', dest='patch', action='store', default=128, type=int,
                        help='patch size')
    parser.add_argument('--dir', dest='dir', action='store', default='./data',
                        help='root directory with images and training data')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    try:
        develop_image(args.camera, args.nip, args.patch, args.image, args.dir)
    except Exception as error:
        log.error(error)


if __name__ == "__main__":
    main()
