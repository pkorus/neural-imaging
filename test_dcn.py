#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import tqdm
import imageio
from pathlib import Path

from compression.afi import dcn_simulate_compression

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr

from models import compression
from helpers import plotting, dataset, coreutils, loading
from compression import jpeg_helpers, afi

supported_plots = ['batch', 'jpeg-match', 'trade-off']


def restore_model(dir_name, patch_size=None):

    with open(os.path.join(dir_name, 'progress.json')) as f:
        progress = json.load(f)
        parameters = progress['dcn']['args']

    parameters['patch_size'] = patch_size
    parameters['default_val_is_train'] = False
    model = compression.AutoencoderDCN(None, None, None, **parameters)    
    model.load_model(dir_name)
    print('Loaded model: {}'.format(model.model_code))

    return model


def match_jpeg(model, batch_x):

    # Compress and decompress model

    batch_y, afi_bytes = afi.dcn_simulate_compression(model, batch_x)

    ssim_dcn = compare_ssim(batch_x.squeeze(), batch_y.squeeze(), multichannel=True)
    bpp_dcn = afi_bytes / np.prod(batch_x.shape[1:-1])

    jpeg_quality = jpeg_helpers.match_ssim(batch_x.squeeze(), ssim_dcn)
    batch_j, bytes_jpeg = jpeg_helpers.compress_batch(batch_x, jpeg_quality)
    ssim_jpeg = compare_ssim(batch_x.squeeze(), batch_y.squeeze(), multichannel=True)
    bpp_jpg = bytes_jpeg / np.prod(batch_x.shape[1:-1])

    fig, axes = plotting.sub(3, ncols=3)
    fig.set_size_inches(12, 10)

    plotting.quickshow(batch_x, 'Original', axes=axes[0])
    plotting.quickshow(batch_y, 'DCN ssim:{:.2f} bpp:{:.2f}'.format(ssim_dcn, bpp_dcn), axes=axes[1])
    plotting.quickshow(batch_j, 'JPEG {} ssim:{:.2f} bpp:{:.2f}'.format(jpeg_quality, ssim_jpeg, bpp_jpg), axes=axes[2])

    fig.tight_layout()
    plt.show()
    plt.close()


def show_example(model, batch_x):

    # Compress and decompress model
    batch_z = model.compress(batch_x)
    batch_y = model.decompress(batch_z)

    # Get empirical histogram of the latent representation
    codebook = model.sess.run(model.codebook).reshape((-1,))

    qmin = np.floor(codebook[0])
    qmax = np.ceil(codebook[-1])    
    
    bin_centers = np.arange(qmin - 1, qmax + 1, 0.1)
    bin_boundaries = np.convolve(bin_centers, [0.5, 0.5], mode='valid')
    bin_centers = bin_centers[1:-1]

    hist_emp = np.histogram(batch_z.reshape((-1,)), bins=bin_boundaries, density=True)[0]
    hist_emp = np.maximum(hist_emp, 1e-9)
    hist_emp = hist_emp / hist_emp.sum()

    # Get TF histogram estimate based on soft quantization
    hist = model.get_tf_histogram(batch_x)

    # Entropy
    entropy = - np.sum(hist * np.log2(hist))
    entropy_emp = - np.sum(hist_emp * np.log2(hist_emp))

    fig, axes = plotting.sub(2, ncols=1)
    fig.set_size_inches(12, 10)
    
    axes[0].plot(bin_centers, hist_emp / hist_emp.max(), 'r-')
    axes[0].plot(codebook, hist / hist.max(), '-bo')

    axes[0].legend(['Empirical H={:.2f}'.format(entropy_emp), 'TF estimate (soft) H={:.2f}'.format(entropy)])
    axes[0].set_ylabel('normalized frequency')
    axes[0].set_xlabel('latent values')
    
    # Thumbnails
    indices = np.argsort(np.var(batch_x, axis=(1, 2, 3)))[::-1]
    thumbs_pairs_few = np.concatenate((batch_x[indices], batch_y[indices]), axis=0)
    thumbs_few = (255 * plotting.thumbnails(thumbs_pairs_few, n_cols=len(batch_x))).astype(np.uint8)

    ssim_values = [compare_ssim(batch_x[i], batch_y[i], multichannel=True) for i in range(len(batch_x))]

    plotting.quickshow(thumbs_few, 'Sample reconstructions, ssim={:.3f}'.format(np.mean(ssim_values)), axes=axes[1])

    fig.tight_layout()
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('plot', help='Plot type ({})'.format(', '.join(supported_plots)))
    # Parameters related to the training data
    parser.add_argument('--data', dest='data', action='store', default='./data/compression/',
                        help='directory with training & validation images (png)')
    parser.add_argument('--images', dest='images', action='store', default=10, type=int,
                        help='number of images to test')
    parser.add_argument('--image', dest='image_id', action='store', default=1, type=int,
                        help='ID of the image to load')
    parser.add_argument('--patch', dest='patch_size', action='store', default=128, type=int,
                        help='training patch size')

    # Parameters of the DCN
    parser.add_argument('--dcn', dest='dcn', action='store', help='specific DCN class name', default='AutoencoderDCN')

    # General
    parser.add_argument('--dir', dest='dir', action='store',
                        help='directory with saved DCN models')

    args = parser.parse_args()

    if not args.dcn:
        print('A DCN needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    # Match the current
    args.plot = coreutils.match_option(args.plot, supported_plots)

    if args.plot == 'batch':

        model = restore_model(args.dir, args.patch_size)

        data = dataset.IPDataset(args.data, load='y', n_images=0, v_images=args.images, val_rgb_patch_size=args.patch_size)
        batch_x = data.next_validation_batch(0, args.images)

        show_example(model, batch_x)

    if args.plot == 'jpeg-match':

        files, _ = loading.discover_files(args.data, n_images=-1, v_images=0)
        files = files[args.image_id:args.image_id+1]
        batch_x = loading.load_images(files, args.data, load='y')
        batch_x = batch_x['y'].astype(np.float32) / (2**8 - 1)

        model = restore_model(args.dir, batch_x.shape[1])

        match_jpeg(model, batch_x)

    if args.plot == 'trade-off':

        # Discover test files
        files, _ = loading.discover_files(args.data, n_images=-1, v_images=0)
        batch_x = loading.load_images(files, args.data, load='y')
        batch_x = batch_x['y'].astype(np.float32) / (2**8 - 1)

        # Get trade-off for JPEG
        quality_levels = np.arange(95, 5, -5)
        df_jpeg_path = os.path.join(args.data, 'jpeg.csv')

        if os.path.isfile(df_jpeg_path):
            print('Restoring JPEG stats from {}'.format(df_jpeg_path))
            df = pd.read_csv(df_jpeg_path, index_col=False)
        else:
            df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'bytes', 'bpp'])

            with tqdm.tqdm(total=len(files) * len(quality_levels), ncols=120, desc='JPEG') as pbar:

                for image_id, filename in enumerate(files):

                    # Read the original image
                    image = batch_x[image_id]

                    for qi, q in enumerate(quality_levels):

                        # Compress images and get effective bytes (only image data - no headers)
                        image_compressed, image_bytes = jpeg_helpers.compress_batch(image, q, effective=True)

                        image_dir = os.path.join(args.data, os.path.splitext(filename)[0])
                        if not os.path.isdir(image_dir):
                            os.makedirs(image_dir)

                        image_path = os.path.join(image_dir, 'jpeg_q{:03d}.png'.format(q))

                        imageio.imwrite(image_path, (255*image_compressed).astype(np.uint8))

                        df = df.append({'image_id': image_id,
                                        'filename': filename,
                                        'codec': 'jpeg',
                                        'quality': q,
                                        'ssim': compare_ssim(image, image_compressed, multichannel=True),
                                        'psnr': compare_psnr(image, image_compressed, data_range=1),
                                        'bytes': image_bytes,
                                        'bpp': 8 * image_bytes / image.shape[0] / image.shape[1]
                                        }, ignore_index=True)

                        pbar.set_postfix(image_id=image_id, quality=q)
                        pbar.update(1)

            df.to_csv(os.path.join(args.data, 'jpeg.csv'), index=False)

        print(df)

        # Discover available models
        model_dirs = list(Path(args.dir).glob('**/progress.json'))
        print('Found {} models'.format(len(model_dirs)))

        for model_dir in model_dirs:
            print('Processing: {}'.format(model_dir))
            dcn = restore_model(os.path.split(str(model_dir))[0], batch_x.shape[1])

            # Dump compressed images
            for image_id, filename in enumerate(files):
                print('.', end='')

                batch_y, image_bytes = afi.dcn_simulate_compression(dcn, batch_x[image_id:image_id + 1])

                # Save the image
                image_dir = os.path.join(args.data, os.path.splitext(filename)[0])
                if not os.path.isdir(image_dir):
                    os.makedirs(image_dir)

                image_path = os.path.join(image_dir, dcn.model_code.replace('/', '-') + '.png')

                imageio.imwrite(image_path, (255*batch_y[0]).astype(np.uint8))

                df = df.append({'image_id': image_id,
                                'filename': filename,
                                'codec': dcn.model_code,
                                'quality': dcn.n_latent,
                                'ssim': compare_ssim(batch_x[image_id], batch_y[0], multichannel=True),
                                'psnr': compare_psnr(batch_x[image_id], batch_y[0], data_range=1),
                                'bytes': image_bytes,
                                'bpp': 8 * image_bytes / batch_x[image_id].shape[0] / batch_x[image_id].shape[1]
                                }, ignore_index=True)
            print('')

        print(df.to_string())
        df.to_csv(os.path.join(args.data, 'dcn.csv'), index=False)


if __name__ == "__main__":
    main()
