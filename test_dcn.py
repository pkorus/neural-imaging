#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt

from models import compression
from helpers import plotting, dataset


def restore_model(dir_name):

    with open(os.path.join(dir_name, 'progress.json')) as f:
        progress = json.load(f)
        parameters = progress['dcn']['args']
    
    model = compression.AutoencoderDCN(None, None, None, **parameters)    
    model.load_model(dir_name)
    print('Loaded model: {}'.format(model.model_code))

    return model


def test_dcn(model, batch_x):

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

    plotting.quickshow(thumbs_few, 'Sample reconstructions', axes=axes[1])

    fig.tight_layout()
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')

    # Parameters related to the training data
    parser.add_argument('--data', dest='data', action='store', default='./data/compression/',
                        help='directory with training & validation images (png)')
    parser.add_argument('--images', dest='images', action='store', default=10, type=int,
                        help='number of images to test')
    parser.add_argument('--patch', dest='patch_size', action='store', default=128, type=int,
                        help='training patch size')

    # Parameters of the DCN
    parser.add_argument('--dcn', dest='dcn', action='store', help='specific DCN class name', default='AutoencoderDCN')

    # General
    parser.add_argument('--dir', dest='dir', action='store',
                        help='output directory with saved model')

    args = parser.parse_args()

    if not args.dcn:
        print('A DCN needs to be specified!')
        parser.print_usage()
        sys.exit(1)

    model = restore_model(args.dir)

    data = dataset.IPDataset(args.data, load='y', n_images=0, v_images=args.images, val_rgb_patch_size=args.patch_size)

    batch_x = data.next_validation_batch(0, args.images)

    test_dcn(model, batch_x)


if __name__ == "__main__":
    main()
