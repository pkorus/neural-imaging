# -*- coding: utf-8 -*-
"""
Provides a Dataset class that loads full resolution training images and samples from them randomly. (See class docs.)
"""
import os
import numpy as np
from helpers import loading
from helpers.loading import sample_patch

from loguru import logger


class Dataset(object):

    def __init__(self, data_directory, *, randomize=2468, load='xy', n_images=120, v_images=30, val_rgb_patch_size=128,
                 val_n_patches=1, val_discard='flat-aggressive'):
        """
        Represents a [RAW-]RGB dataset for training imaging pipelines. The class preloads full resolution images and
        samples from them when requesting training batches. (Validation images are sampled upon creation.) Patch
        selection takes care of proper alignment between RAW images (represented as half-size 4-channel RGGB stacks) and
        their corresponding rendered RGB versions. Selection can be controlled to prefer certain types of patches (not
        strictly enforced). The following DISCARD modes are available:

        - flat - attempts to discard flat patches based on patch variance (not strict)
        - flat-aggressive - a more aggressive version that avoids patches with variance < 0.01
        - dark-n-textured - avoid dark (mean < 0.35) and textured patches (variance > 0.005)

        Usage examples:
        ---------------

        # Load a RAW -> RGB dataset
        data = Dataset('data/raw/training_data/D90')
        batch_raw, batch_rgb = data.next_training_batch(0, 10, 128, 'flat-aggressive')

        # Load RGB only dataset
        data = Dataset('data/rgb/native12k/', load='y')
        batch_rgb = data.next_training_batch(0, 10, 128, 'flat-aggressive')

        :param data_directory: directory path with RAW-RGB pairs (*.npy & *.png) or only RGB images (*.png)
        :param randomize: randomization seed
        :param load: what data to load: 'xy' load RAW+RGB, 'x' load RAW only, 'y' load RGB only
        :param n_images: number of training images (full resolution)
        :param v_images: number of validation images (patches sampled upon creation)
        :param val_rgb_patch_size: validation patch size
        :param val_n_patches: number of validation patches to load per full-resolution image
        :param val_discard: patch discard mode (for validation data)
        """

        if not any(load == allowed for allowed in ['xy', 'x', 'y']):
            raise ValueError('Invalid X/Y data requested!')

        if not os.path.isdir(data_directory):
            if '/' in data_directory or '\\' in data_directory:
                raise ValueError(f'Cannot find the data directory: {data_directory}')

            if os.path.isdir(os.path.join('data/raw/training_data/', data_directory)):
                data_directory = os.path.join('data/raw/training_data/', data_directory)
            elif os.path.isdir(os.path.join('data/rgb/', data_directory)):
                data_directory = os.path.join('data/rgb/', data_directory)
            else:
                raise ValueError(f'Cannot find the data directory: {data_directory}')

        self.files = {}
        self._loaded_data = load
        self._data_directory = data_directory
        self._counts = (n_images, v_images, val_n_patches)
        self._val_discard = 'flat-aggressive'
        self.files['training'], self.files['validation'] = loading.discover_images(data_directory, randomize=randomize,
                                                                                   n_images=n_images, v_images=v_images)

        self.data = {
            'training': loading.load_images(self.files['training'], data_directory, load=load),
            'validation': loading.load_patches(self.files['validation'], data_directory,
                                               patch_size=val_rgb_patch_size // 2, n_patches=val_n_patches,
                                               load=load, discard=val_discard)
        }

        if 'y' in self.data['training']:
            self.H, self.W = self.data['training']['y'].shape[1:3]
        else:
            self.H, self.W = (2 * dim for dim in self.data['training']['x'].shape[1:3])

    def __getitem__(self, key):
        if key in ['training', 'validation']:
            return self.data[key]
        else:
            raise KeyError('Key: {} not found!'.format(key))

    def next_training_batch(self, batch_id, batch_size, rgb_patch_size, discard='flat', max_attempts=25):
        """
        Sample a new batch of training patches.
        :param batch_id: integer from 0 to (#training images // batch_size - 1)
        :param batch_size: integer, self explanatory
        :param rgb_patch_size: patch size (in full-resolution RGB coordinates; RAW patches [RGGB] have half the size)
        :param discard: patch discard mode (for validation data)
        :param max_attempts: maximum number of sampling attempts (if unsuccessful)
        :return: tuple of np arrays (RAW, RGB) or np array (RGB)
        """

        if discard is not None and 'y' not in self.data['training']:
            raise ValueError('Cannot discard patches if RGB data is not loaded.')

        if (batch_id + 1) * batch_size > len(self.files['training']):
            raise ValueError('Not enough images for the requested batch_id & batch_size')

        raw_patch_size = rgb_patch_size // 2

        # Allocate memory for the batch
        batch = {
            'x': np.zeros((batch_size, raw_patch_size, raw_patch_size, 4), dtype=np.float32) if 'x' in self._loaded_data else None,
            'y': np.zeros((batch_size, rgb_patch_size, rgb_patch_size, 3), dtype=np.float32) if 'y' in self._loaded_data else None
        }

        for b in range(batch_size):

            bid = batch_id * batch_size + b
            current_rgb = self.data['training']['y'][bid]
            xx, yy = sample_patch(current_rgb, rgb_patch_size, discard, max_attempts)
            rx, ry = xx // 2, yy // 2

            if 'x' in self._loaded_data:
                current_raw = self.data['training']['x'][bid]
                batch['x'][b] = current_raw[ry:ry+raw_patch_size, rx:rx+raw_patch_size].astype(np.float) / (2**16 - 1)
            if 'y' in self._loaded_data:
                batch['y'][b] = current_rgb[yy:yy+rgb_patch_size, xx:xx+rgb_patch_size].astype(np.float) / (2**8 - 1)

        if self._loaded_data == 'xy':
            return batch['x'], batch['y']
        elif self._loaded_data == 'y':
            return batch['y']
        elif self._loaded_data == 'x':
            return batch['x']

    def next_validation_batch(self, batch_id, batch_size):
        """
        Return a validation batch.
        :param batch_id: integer from 0 to (#validation images // batch_size - 1)
        :param batch_size: integer, self explanatory
        :return: tuple of np arrays (RAW, RGB) or np array (RGB)
        """
        rgb_patch = self.rgb_patch_size

        batch = {
            'x': np.zeros((batch_size, rgb_patch // 2, rgb_patch // 2, 4), dtype=np.float32) if 'x' in self._loaded_data else None,
            'y': np.zeros((batch_size, rgb_patch, rgb_patch, 3), dtype=np.float32) if 'y' in self._loaded_data else None
        }

        for b in range(batch_size):
            if 'x' in self._loaded_data:
                batch['x'][b] = self.data['validation']['x'][batch_id * batch_size + b].astype(np.float) / (2 ** 16 - 1)
            if 'y' in self._loaded_data:
                batch['y'][b] = self.data['validation']['y'][batch_id * batch_size + b].astype(np.float) / (2 ** 8 - 1)

        if self._loaded_data == 'xy':
            return batch['x'], batch['y']
        elif self._loaded_data == 'y':
            return batch['y']
        elif self._loaded_data == 'x':
            return batch['x']

    def is_raw_and_rgb(self):
        return len(self._loaded_data) == 2

    @property
    def rgb_patch_size(self):
        if 'y' in self._loaded_data:
            patch_size = self.data['validation']['y'].shape[1]
        else:
            patch_size = 2 * self.data['validation']['x'].shape[1]
        return patch_size

    @property
    def count_training(self):
        key = self._loaded_data[0]
        return self.data['training'][key].shape[0]

    @property
    def count_validation(self):
        key = self._loaded_data[0]
        return self.data['validation'][key].shape[0]

    def __repr__(self):
        args = [f'"{self._data_directory}"', f'load="{self._loaded_data}"', f'n_images={self._counts[0]}',
                f'v_images={self._counts[1]}', f'val_rgb_patch_size={self._counts[2]}',
                f'val_rgb_patch_size={self.rgb_patch_size}', f'discard="{self._val_discard}"']
        return f'Dataset({", ".join(args)})'

    def shapes(self):
        stats = {
            'path': self._data_directory,
        }

        for k in self._loaded_data:
            stats['training/{}'.format(k)] = self.data['training'][k].shape
            stats['validation/{}'.format(k)] = self.data['validation'][k].shape

        return stats

    @property
    def loaded_data(self):
        if self._loaded_data == 'xy':
            db_type = 'raw+rgb'
        elif self._loaded_data == 'y':
            db_type = 'rgb'
        elif self._loaded_data == 'x':
            db_type = 'raw'
        return db_type

    def summary(self):
        valid_label = '' if self._val_discard is None else f', {self._val_discard}'
        return f'Dataset[{os.path.split(self._data_directory)[-1]},{self.loaded_data}] : {self.count_training} train. images + {self.count_validation} valid. patches ({self.rgb_patch_size} px{valid_label})'

    def details(self):
        label = [self.summary()]

        for k, l in zip('xy', ['RAW', 'RGB']):
            if k in self._loaded_data:
                label.append(f'{l} -> training {self.data["training"][k].shape} + validation {self.data["validation"][k].shape}')

        return '\n'.join(label)

    def get_training_generator(self, batch_size, rgb_patch_size, discard='flat'):
        """
        Get a generator for training data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_training_generator(batch_size, rgb_patch_size, discard),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """

        for batch_id in range(self.count_training // batch_size):
            yield self.next_training_batch(batch_id, batch_size, rgb_patch_size, discard)

        raise StopIteration()

    def get_validation_generator(self, batch_size):
        """
        Get a generator for validation data. Can be used to construct a data pipeline:

        dp = tf.data.Dataset.from_generator(lambda: data.get_validation_generator(batch_size),
            output_types=len(self._loaded_data) * (tf.float32, ))
        """

        for batch_id in range(self.count_validation // batch_size):
            yield self.next_validation_batch(batch_id, batch_size)

        raise StopIteration()

