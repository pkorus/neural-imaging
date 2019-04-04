#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse
import tensorflow as tf

from helpers import coreutils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('data')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


extensions = '(npy)'
raw_extensions = ['.nef', '.dng', '.NEF', '.DNG']
supported_pipelines = ['libRAW', 'Python', 'INet', 'DNet', 'UNet']


def develop_images(camera, pipeline, n_images=0, root_dir='./data/raw/', model_dir='nip_model_snapshots', dev_dir='nip_developed', nip_params=None):

    if pipeline not in supported_pipelines:
        raise ValueError('Unsupported pipeline model ({})! Available models: {}'.format(pipeline, ', '.join(supported_pipelines)))

    dir_models = os.path.join(root_dir, model_dir)
    nip_directory = os.path.join(root_dir, 'nip_training_data', camera)
    out_directory = os.path.join(root_dir, dev_dir, camera, pipeline)
    raw_directory = os.path.join(root_dir, 'images', camera)

    if not os.path.exists(nip_directory):
        raise IOError('Directory not found! {}'.format(nip_directory))

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    # Lazy loading of remaining dependencies to ensure responsiveness of the CLI
    import numpy as np
    import imageio
    import tqdm
    from helpers import raw_api
    from models import pipelines

    print('Camera: {}'.format(camera))
    print('Pipeline: {}'.format(pipeline))
    print('NIP Models: {}'.format(dir_models))
    print('NIP Training Directory: {}'.format(nip_directory))
    print('Out Directory: {}'.format(out_directory))

    # %% Process Bayer stacks with the given pipeline
    npy_filenames = coreutils.listdir(nip_directory, '.*\.{}$'.format(extensions))
    log.info('Camera {} matched {:,} Bayer stacks'.format(camera, len(npy_filenames)))

    manual_dev_settings = {'use_srgb': True, 'use_gamma': True, 'brightness': None}

    # Setup the NIP model
    if pipeline.endswith('Net'):
        sess = tf.Session()
        model = getattr(pipelines, pipeline)(sess, tf.get_default_graph(), loss_metric='L2', **nip_params)
        model.load_model(camera, out_directory_root=dir_models)

    # Limit the number of images
    if n_images > 0:
        npy_filenames = npy_filenames[:n_images]

    for npy_file in tqdm.tqdm(npy_filenames, ncols=120, desc='Developing ({}/{})'.format(camera, pipeline)):

        # Find the original RAW file (for standard pipelines.py)
        raw_file = os.path.join(raw_directory, os.path.splitext(npy_file)[0])
        raw_found = False

        for extension in raw_extensions:
            if os.path.exists(raw_file + extension):
                raw_file = raw_file + extension
                raw_found = True
                break

        if not raw_found:
            raise RuntimeError('RAW file not found for Bayer stack: {}'.format(npy_file))

        out_png = os.path.join(out_directory, os.path.splitext(npy_file)[0] + '.png')

        if not os.path.exists(out_png):
            # Process with the desired pipeline
            if pipeline == 'libRAW':
                rgb = raw_api.process_auto(raw_file)
            elif pipeline == 'Python':
                rgb = 255 * raw_api.process(raw_file, **manual_dev_settings)
                rgb = rgb.astype(np.uint8)
            else:
                # Find the cached Bayer stack
                bayer_file = os.path.join(nip_directory, npy_file)
                bayer_stack = np.load(bayer_file).astype(np.float32) / (2**16 - 1)
                rgb = 255 * model.process(bayer_stack).squeeze()
                rgb = rgb.astype(np.uint8)

            imageio.imwrite(out_png, rgb.astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--pipe', dest='pipeline', action='store', default='libRAW',
                        help='imaging pipeline ({})'.format(supported_pipelines))
    parser.add_argument('--dir', dest='dir', action='store', default='./data/raw/',
                        help='root directory with images and training data')
    parser.add_argument('--model_dir', dest='model_dir', action='store', default='nip_model_snapshots',
                        help='directory with TF models')                        
    parser.add_argument('--dev_dir', dest='dev_dir', action='store', default='nip_developed',
                        help='output directory')
    parser.add_argument('--params', dest='nip_params', default=None, help='Extra parameters for NIP constructor (JSON string)')    
    parser.add_argument('--images', dest='images', action='store', default=0, type=int,
                        help='number of images to process')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    try:
        if args.nip_params is not None:
            args.nip_params = json.loads(args.nip_params.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.nip_params.replace('\'', '"'))
        sys.exit(2)

    try:
        develop_images(args.camera, args.pipeline, args.images, args.dir, args.model_dir, args.dev_dir, nip_params=args.nip_params)
    except Exception as error:
        log.error(error)


if __name__ == "__main__":
    main()
