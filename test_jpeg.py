#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import imageio as io
import argparse
from helpers import plots
from skimage.measure import compare_psnr
from matplotlib import pylab as plt
from models.jpeg import DJPG

# Disable unimportant logging and import TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DEFAULT_IMAGE = 'docs/schematic_overview.png'


def test_output(image, jpeg_quality=50, rounding_approximation=None):

    jpg = DJPG(rounding_approximation=rounding_approximation)
    print(jpg)

    batch_x = np.expand_dims(image, 0)
    batch_y = jpg.process(batch_x / 255, jpeg_quality)

    n_images = batch_x.shape[0]

    batch_j = np.zeros_like(batch_x)
    for n in range(n_images):
        io.imwrite('/tmp/patch_{}.jpg'.format(n), (batch_x[n].squeeze()).astype(np.uint8), quality=jpeg_quality, subsampling='4:4:4')
        batch_j[n] = io.imread('/tmp/patch_{}.jpg'.format(n))

    for n in range(n_images):
        plt.subplot(n_images, 3, 1 + n*3)
        plots.image(batch_x[n].squeeze() / np.max(np.abs(batch_x)), 'Input')

        plt.subplot(n_images, 3, 2 + n*3)
        plots.image(batch_y[n].squeeze() / np.max(np.abs(batch_y)), 'dJPEG Model')

        plt.subplot(n_images, 3, 3 + n*3)
        plots.image(batch_j[n].squeeze() / np.max(np.abs(batch_j)), 'libJPG Codec')

    plt.show()


def test_quality(image, rounding_approximation=None, n_quality_levels=91):

    jpg = DJPG(rounding_approximation=rounding_approximation)
    print(jpg)

    batch_x = np.expand_dims(image[0:1024, 0:1024, :], 0)

    psnrs_y, psnrs_j = [], []

    quality_levels = np.unique(np.round(np.linspace(10, 100, n_quality_levels)).astype(np.int32)).tolist()
    print('Using quality levels: {}'.format(quality_levels))

    for jpeg_quality in quality_levels:
        batch_y = jpg.process(batch_x / 255, jpeg_quality)
        batch_y = np.round(255 * batch_y) / 255
        io.imwrite('/tmp/patch.jpg', (batch_x.squeeze()).astype(np.uint8), quality=jpeg_quality, subsampling='4:4:4')
        batch_j = io.imread('/tmp/patch.jpg')
        psnrs_y.append(compare_psnr(batch_x.squeeze(), 255 * batch_y.squeeze(), 255))
        psnrs_j.append(compare_psnr(batch_x.squeeze(), batch_j.squeeze(), 255))

    # Plot
    plt.figure(figsize=(6,6))
    plt.plot(psnrs_y, psnrs_j, 'bo', alpha=0.25)
    plt.plot([30, 50], [30, 50], 'k:')
    plt.xlabel('PSNR for dJPEG')
    plt.ylabel('PSNR for libJPEG')
    plt.xlim([30, 60])
    plt.ylim([30, 50])
    if rounding_approximation is None:
        plt.title('dJPEG vs libJPEG quality (with standard rounding)'.format(rounding_approximation))
    else:
        plt.title('dJPEG vs libJPEG quality (with {} rounding approx.)'.format(rounding_approximation))

    for i, q in enumerate(quality_levels):
        if q % 10 == 0:
            plt.plot(psnrs_y[i], psnrs_j[i], 'ko')
            plt.text(psnrs_y[i]+1, psnrs_j[i]-0.25, 'Q{:02d}'.format(q))
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test the dJPEG model')
    parser.add_argument('mode', help='Test mode: output / quality')
    parser.add_argument('--image', dest='image', action='store',
                        help='test image path')
    parser.add_argument('--patch', dest='patch_size', action='store', type=int, default=256,
                        help='patch size (default 256)')
    parser.add_argument('--quality', dest='quality', action='store', type=int, default=50,
                        help='the quality level or number of levels for evaluation')
    parser.add_argument('--round', dest='round', action='store', default='soft',
                        help='rounding approximation mode: sin, soft, harmonic')

    args = parser.parse_args()
    
    args.image = args.image or DEFAULT_IMAGE
    
    if not os.path.exists(args.image):
        print('Error: file does not exist! {}'.format(args.image))

    image = io.imread(args.image)

    if image.shape[0] > args.patch_size or image.shape[1] > args.patch_size:
        xx = (image.shape[1] - args.patch_size) // 2
        yy = (image.shape[0] - args.patch_size) // 2
        image = image[yy:yy+args.patch_size, xx:xx+args.patch_size, :]

    print('Using image: {}x{} px'.format(*image.shape[:2]))

    if args.mode == 'output':
        test_output(image, jpeg_quality=int(args.quality), rounding_approximation=args.round)

    elif args.mode == 'quality':
        test_quality(image, n_quality_levels=int(args.quality), rounding_approximation=args.round)


if __name__ == "__main__":
    main()
