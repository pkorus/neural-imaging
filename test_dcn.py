#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

from helpers import plotting, dataset, coreutils, loading, utils
from compression import jpeg_helpers, afi, ratedistortion

supported_plots = ['batch', 'jpeg-match', 'jpg-trade-off', 'jp2-trade-off', 'dcn-trade-off']


def match_jpeg(model, batch_x, axes=None):

    # Compress using DCN and get number of bytes
    batch_y, bytes_dcn = afi.dcn_simulate_compression(model, batch_x)

    ssim_dcn = compare_ssim(batch_x.squeeze(), batch_y.squeeze(), multichannel=True, data_range=1)
    bpp_dcn = 8 * bytes_dcn / np.prod(batch_x.shape[1:-1])

    try:
        jpeg_quality = jpeg_helpers.match_ssim(batch_x.squeeze(), ssim_dcn)
    except:
        if ssim_dcn > 0.8:
            jpeg_quality = 95
        else:
            jpeg_quality = 10
        print('WARNING Could not find a matching JPEG quality factor - guessing {}'.format(jpeg_quality))

    # Compress using JPEG
    batch_j, bytes_jpeg = jpeg_helpers.compress_batch(batch_x[0], jpeg_quality, effective=True)
    ssim_jpeg = compare_ssim(batch_x.squeeze(), batch_y.squeeze(), multichannel=True, data_range=1)
    bpp_jpg = 8 * bytes_jpeg / np.prod(batch_x.shape[1:-1])

    # Get stats
    code_book = model.get_codebook()
    batch_z = model.compress(batch_x)
    counts = utils.qhist(batch_z, code_book)
    counts = counts.clip(min=1)
    probs = counts / counts.sum()
    entropy = - np.sum(probs * np.log2(probs))

    # Print report
    print('DCN             : {}'.format(model.model_code))
    print('Pixels          : {}x{} = {:,} px'.format(batch_x.shape[1], batch_x.shape[2], np.prod(batch_x.shape[1:-1])))
    print('Bitmap          : {:,} bytes'.format(np.prod(batch_x.shape)))
    print('Code-book size  : {} elements from {} to {}'.format(len(code_book), min(code_book), max(code_book)))
    print('Entropy         : {:.2f} bits per symbol'.format(entropy))
    print('Latent size     : {:,}'.format(np.prod(batch_z.shape)))
    print('PPF Naive       : {:,.0f} --> {:,.0f} bytes [{} bits per element]'.format(
        np.prod(batch_z.shape) * np.log2(len(code_book)) / 8,
        np.prod(batch_z.shape) * np.ceil(np.log2(len(code_book))) / 8,
        np.ceil(np.log2(len(code_book)))
    ))
    print('PPF Theoretical : {:,.0f} bytes ({:.2f} bpp)'.format(
        np.prod(batch_z.shape) * entropy / 8,
        np.prod(batch_z.shape) * entropy / np.prod(batch_x.shape[1:-1])))
    print('FSE Coded       : {:,} bytes ({:.2f} bpp) --> ssim: {:.3f}'.format(bytes_dcn, bpp_dcn, ssim_dcn))
    print('JPEG (Q={:2d})     : {:,} bytes ({:0.2f} bpp) --> ssim: {:.3f} // effective size disregarding JPEG headers'.format(jpeg_quality, bytes_jpeg, bpp_jpg, ssim_jpeg))

    # Plot results
    if axes is None:
        fig, axes = plotting.sub(6, ncols=3)
        fig.set_size_inches(12, 10)
        fig.tight_layout()
    else:
        fig = axes[0].figure

    # Plot full-resolution
    plotting.quickshow(batch_x, 'Original ({0}x{0})'.format(batch_x.shape[1]), axes=axes[0])
    plotting.quickshow(batch_y, 'DCN ssim:{:.2f} bpp:{:.2f}'.format(ssim_dcn, bpp_dcn), axes=axes[1])
    plotting.quickshow(batch_j, 'JPEG {} ssim:{:.2f} bpp:{:.2f}'.format(jpeg_quality, ssim_jpeg, bpp_jpg), axes=axes[2])

    # Plot zoom
    crop_size = max([64, batch_x.shape[1] // 4])
    plotting.quickshow(utils.crop_middle(batch_x, crop_size), 'Original crop ({0}x{0})'.format(crop_size), axes=axes[3])
    plotting.quickshow(utils.crop_middle(batch_y, crop_size), 'DCN crop ({0}x{0})'.format(crop_size), axes=axes[4])
    plotting.quickshow(utils.crop_middle(batch_j, crop_size), 'JPEG crop ({0}x{0})'.format(crop_size), axes=axes[5])

    return fig


def show_example(model, batch_x):

    # Compress and decompress model
    batch_z = model.compress(batch_x)
    batch_y = model.decompress(batch_z)

    # Get empirical histogram of the latent representation
    codebook = model.get_codebook()

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
    return fig


def main():
    parser = argparse.ArgumentParser(description='Test a neural imaging pipeline')
    parser.add_argument('plot', help='Plot type ({})'.format(', '.join(supported_plots)))
    # Parameters related to the training data
    parser.add_argument('--data', dest='data', action='store', default='./data/clic/',
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

        model, stats = afi.restore_model(args.dir, args.patch_size, fetch_stats=True)
        print('Training stats:', stats)

        data = dataset.IPDataset(args.data, load='y', n_images=0, v_images=args.images, val_rgb_patch_size=args.patch_size)
        batch_x = data.next_validation_batch(0, args.images)

        fig = show_example(model, batch_x)
        plt.show()
        plt.close()

    elif args.plot == 'jpeg-match':

        files, _ = loading.discover_files(args.data, n_images=-1, v_images=0)
        files = files[args.image_id:args.image_id+1]
        batch_x = loading.load_images(files, args.data, load='y')
        batch_x = batch_x['y'].astype(np.float32) / (2**8 - 1)

        model = afi.restore_model(args.dir, batch_x.shape[1])

        fig = match_jpeg(model, batch_x)
        plt.show()
        plt.close()

    elif args.plot == 'jpg-trade-off':

        df = ratedistortion.get_jpeg_df(args.data, write_files=True)
        print(df.to_string())

    elif args.plot == 'jp2-trade-off':

        df = ratedistortion.get_jpeg2k_df(args.data, write_files=True)
        print(df.to_string())

    elif args.plot == 'dcn-trade-off':

        df = ratedistortion.get_dcn_df(args.data, args.dir, write_files=True)
        print(df.to_string())
    else:
        print('Error: Unknown plot!')


if __name__ == "__main__":
    main()
