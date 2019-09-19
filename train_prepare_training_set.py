#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import imageio
import exifread
import os
import sys
import logging
import tqdm
import argparse
from helpers import raw_api, coreutils

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('data')

EXTENSIONS = '(NEF|DNG|CR2|AWR)'


def prepare_training_set(camera, target_pipeline, dev_settings, n_images=150, root_dir='./data/'):

    if target_pipeline not in ['auto', 'manual']:
        raise ValueError('Unsupported target pipeline!')

    raw_directory = os.path.join(root_dir, 'raw', 'images', camera)
    out_directory = os.path.join(root_dir, 'raw', 'training_data', camera)

    if not os.path.exists(raw_directory):
        log.error('Directory not found! {}'.format(raw_directory))
        sys.exit(2)

    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    print('RAW Directory: {}'.format(raw_directory))
    print('Out Directory: {}'.format(out_directory))

    # List RAW files and find the ones with horizontal orientation
    raw_filenames = coreutils.listdir(raw_directory, '.*\.{}$'.format(EXTENSIONS))
    log.info('Camera {} matched {:,} RAW images'.format(camera, len(raw_filenames)))

    raw_filenames_selected = []

    for nef_file in raw_filenames:

        with open(os.path.join(raw_directory, nef_file), 'rb') as f:
            tags = exifread.process_file(f, details=False, stop_tag='Image Orientation')
            orientation = tags['Image Orientation'].printable
            log.info('{} -> {}'.format(nef_file, orientation))
            if orientation.startswith('Horizontal'):
                raw_filenames_selected.append(nef_file)

        if len(raw_filenames_selected) >= n_images:
            break

    log.info('Collected {} landscape-oriented photos for training'.format(len(raw_filenames_selected)))

    if len(raw_filenames_selected) < n_images:
        log.error('Not enough horizontal images! Found {} but expected {}.'.format(len(raw_filenames_selected), n_images))

    dev_settings = dev_settings or {'use_srgb': True, 'use_gamma': True, 'brightness': None}

    # Iterate over RAW files and produce:
    #  1. RGGB Bayer stacks (H/2, W/2, 4)
    #  2. RGB Optimization target (H, W, 3)
    for nef_file in tqdm.tqdm(raw_filenames_selected, ncols=120, desc='Preparing train. data ({})'.format(camera)):

        out_npy = os.path.join(out_directory, os.path.splitext(nef_file)[0] + '.npy')
        out_png = os.path.join(out_directory, os.path.splitext(nef_file)[0] + '.png')

        try:
            if not os.path.exists(out_npy):
                image_bayer = raw_api.stacked_bayer(os.path.join(raw_directory, nef_file), use_wb=True)
                image_bayer = ((2**16 - 1) * image_bayer).astype(np.uint16)
                np.save(out_npy, image_bayer)

            if not os.path.exists(out_png):
                if target_pipeline == 'auto':
                    rgb = raw_api.process_auto(os.path.join(raw_directory, nef_file))
                elif target_pipeline == 'manual':
                    rgb = 255 * raw_api.process(os.path.join(raw_directory, nef_file), **dev_settings)
                else:
                    raise ValueError('Unsupported develop mode!')
                imageio.imwrite(out_png, rgb.astype(np.uint8))

        except Exception as error:
            log.error('RAW Processing failed for file: {}'.format(nef_file))
            log.error(error)
            sys.exit(2)

    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='Prepares training pairs (raw inputs in *.npy and optimization targets in *.png) for a given camera')
    parser.add_argument('--cam', dest='camera', action='store', help='camera')
    parser.add_argument('--target', dest='target', action='store', default='manual',
                        help='target for optimization (manual or auto)')
    parser.add_argument('--dir', dest='dir', action='store', default='./data',
                        help='root directory with images and training data')
    parser.add_argument('--images', dest='images', action='store', default=150, type=int,
                        help='number of images to prepare')

    args = parser.parse_args()

    if not args.camera:
        print('A camera needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    prepare_training_set(args.camera, args.target, None, args.images, args.dir)


if __name__ == "__main__":
    main()
