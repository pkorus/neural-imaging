# -*- coding: utf-8 -*-
"""
Helper functions for finding & loading images and extracting patches.
"""
import os
import numpy as np
import tqdm
import imageio
from helpers import fsutil

from loguru import logger


def discover_images(data_directory, n_images=120, v_images=30, extension='png', randomize=0):
    """
    Find available images and split them into training / validation sets.
    :param data_directory: directory
    :param n_images: number of training images
    :param v_images: number of validation images
    :param extension: file extension
    :param randomize: whether to shuffle files before the split
    """

    files = fsutil.listdir(data_directory, '.*\\.{}$'.format(extension))
    logger.debug(f'{data_directory}: in total {len(files)} files available')

    if randomize:
        np.random.seed(randomize)
        np.random.shuffle(files)

    if n_images == 0 and v_images == -1:
        v_images = len(files)

    if n_images == -1 and v_images == 0:
        n_images = len(files)

    if len(files) >= n_images + v_images:
        val_files = files[n_images:(n_images + v_images)]
        files = files[0:n_images]
    else:
        raise ValueError('Not enough images!')
        
    return files, val_files


def load_images(files, data_directory, extension='png', load='xy'):
    """
    Load pairs of full-resolution images: (raw, rgb). Raw inputs are stored in *.npy files (see
    train_prepare_training_set.py).
    :param files: list of files to be loaded
    :param data_directory: directory path
    :param extension: file extension of rgb images
    :param load: what data to load - string: 'xy' (load both raw and rgb), 'x' (load only raw) or 'y' (load only rgb)
    """
    n_images = len(files)

    if n_images == 0:
        logger.warning('No images to load!')
        return {k: np.zeros(shape=(1, 1, 1, 1)) for k in load}
    
    # Check image resolution
    image = imageio.imread(os.path.join(data_directory, files[0]))
    resolutions = (image.shape[0] >> 1, image.shape[1] >> 1)
    del image
    
    data = {}
    
    if 'x' in load:
        data['x'] = np.zeros((n_images, *resolutions, 4), dtype=np.uint16)
    if 'y' in load:
        data['y'] = np.zeros((n_images, 2 * resolutions[0], 2 * resolutions[1], 3), dtype=np.uint8)

    with tqdm.tqdm(total=n_images, ncols=100, desc='Loading images') as pbar:

        for i, file in enumerate(files):
            npy_file = file.replace('.{}'.format(extension), '.npy')

            if 'x' in data:
                data['x'][i] = np.load(os.path.join(data_directory, npy_file))
            if 'y' in data:
                data['y'][i] = imageio.imread(os.path.join(data_directory, file), pilmode='RGB')

            pbar.update(1)

        return data

    
def load_patches(files, data_directory, patch_size=128, n_patches=100, discard='flat-aggressive', extension='png', load='xy'):
    """
    Sample (raw, rgb) pairs or random patches from given images.

    :param files: list of available images
    :param data_directory: directory path
    :param patch_size: patch size (in the raw image - rgb patches will be twice as big)
    :param n_patches: number of patches per image
    :param discard: strategy for discarding nonsuitable patches
    :param extension: file extension of RGB images
    :param load: what data to load - string: 'xy' (load both raw and rgb), 'x' (load only raw) or 'y' (load only rgb)
    """
    v_images = len(files)
    max_attempts = 100
    discard_label = '(random)' if discard is None else '({})'.format(discard)
    data = {}

    if 'x' in load: data['x'] = np.zeros((v_images * n_patches, patch_size, patch_size, 4), dtype=np.uint16)
    if 'y' in load: data['y'] = np.zeros((v_images * n_patches, 2 * patch_size, 2 * patch_size, 3), dtype=np.uint8)

    with tqdm.tqdm(total=v_images * n_patches, ncols=100, desc='Loading patches {}'.format(discard_label)) as pbar:

        for i, file in enumerate(files):
            npy_file = file.replace('.{}'.format(extension), '.npy')

            if 'x' in data: image_x = np.load(os.path.join(data_directory, npy_file))
            if 'y' in data: image_y = imageio.imread(os.path.join(data_directory, file), pilmode='RGB')

            # Sample random patches
            for b in range(n_patches):

                xx, yy = sample_patch(image_y, 2 * patch_size, discard, max_attempts)
                rx, ry = xx // 2, yy // 2

                if 'x' in data:
                    data['x'][i * n_patches + b] = image_x[ry:ry + patch_size, rx:rx + patch_size, :]
                if 'y' in data:
                    data['y'][i * n_patches + b] = image_y[yy:yy + 2*patch_size, xx:xx + 2*patch_size, :]

                pbar.update(1)

        return data


def sample_patch(rgb_image, rgb_patch_size=128, discard=None, max_attempts=25):
    """
    Sample a single patch from a full-resolution image. Sampling can be fully random or can follow a discarding policy.
    The following DISCARD modes are available:

        - flat - attempts to discard flat patches based on patch variance (not strict)
        - flat-aggressive - a more aggressive version that avoids patches with variance < 0.01
        - dark-n-textured - avoid dark (mean < 0.35) and textured patches (variance > 0.005)

    :param rgb_image: full resolution RGB image
    :param rgb_patch_size: integer, self-explanatory
    :param discard: discard policy
    :param max_attempts: maximum number of sampling attempts (if unsuccessful)
    :return: a tuple with (x, y) coordinates
    """
    xx, yy = 0, 0

    max_x = rgb_image.shape[1] - rgb_patch_size
    max_y = rgb_image.shape[0] - rgb_patch_size

    if max_x > 0 or max_y > 0:
        found = False
        panic_counter = max_attempts

        while not found:
            # Sample a random patch - the number needs to be even to ensure proper Bayer alignment
            xx = 2 * (np.random.randint(0, max_x) // 2) if max_x > 0 else 0
            yy = 2 * (np.random.randint(0, max_y) // 2) if max_y > 0 else 0

            if not discard:
                found = True
                continue

            patch = rgb_image[yy:yy + rgb_patch_size, xx:xx + rgb_patch_size].astype(np.float) / 255
            patch_variance = np.var(patch)
            patch_intensity = np.mean(patch)

            # Check if the sampled patch is acceptable
            if discard == 'flat':

                if patch_variance < 0.005:
                    panic_counter -= 1
                    found = False if panic_counter > 0 else True
                elif patch_variance < 0.01:
                    found = np.random.uniform() > 0.5
                else:
                    found = True

            elif discard == 'flat-aggressive':

                if patch_variance < 0.02:
                    if panic_counter == max_attempts or patch_variance > best_patch[-1]:
                        best_patch = (xx, yy, patch_variance)
                    panic_counter -= 1
                    found = False if panic_counter > 0 else True
                    if found:
                        xx, yy, patch_variance = best_patch
                else:
                    found = True

            elif discard == 'dark-n-textured':

                if 0 < patch_variance < 0.005 and 0.35 < patch_intensity < 0.99:
                    found = True
                else:
                    if panic_counter == max_attempts or (patch_variance < 2 * best_patch[-1]
                                                         and patch_intensity > 1.1 * best_patch[-2]):
                        best_patch = (xx, yy, patch_intensity, patch_variance)
                    panic_counter -= 1
                    found = False if panic_counter > 0 else True
                    if found:
                        xx, yy, patch_intensity, patch_variance = best_patch

            elif discard is None:
                found = True

            else:
                raise ValueError('Unrecognized discard mode: {}'.format(discard))

    return xx, yy
