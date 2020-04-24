# -*- coding: utf-8 -*-
"""
Implements JPEG compression models. The module provides the following:

The `DifferentiableJPEG` class is a custom Tensorflow model which manually implements the JPEG codec using basic
matrix operations. In most cases, this model should not be used directly.

The `JPEG` class is the main model for most use cases. It serves as a high-level wrapper over `DifferentiableJPEG` but
provides multiple extra features:
- it interfaces naturally with the other components in the toolbox
- allows to switch JPEG codecs, e.g., switch to libJPEG for final validation
- allows for randomly choose quality factors (useful for data augmentation)

An additional helper function `differentiable_jpeg` can be used if you don't want to keep your own JPEG instance. The
function will expose a lazy-initialized instance of the JPEG model (differentiable approximation).

"""
import numpy as np
import tensorflow as tf

from models.layers import Quantization
from models.tfmodel import TFModel
from compression import jpeg_helpers
from helpers.utils import is_number
from compression.jpeg_helpers import jpeg_qtable, jpeg_qf_estimation

_common_codec = None


def is_valid_quality(quality):
    if is_number(quality) and 1 <= quality <= 100:
        return True
    elif hasattr(quality, '__getitem__') and len(quality) > 1 and all((1 <= x <= 100) for x in quality):
        return True
    return False


def differentiable_jpeg(x, quality):
    global _common_codec
    if _common_codec is None:
        _common_codec = JPEG(None, 'soft')
    return _common_codec.process(x, quality)


class DifferentiableJPEG(tf.keras.Model):

    def __init__(self, quality=None, rounding_approximation='sin', rounding_approximation_steps=5, trainable=False):
        super().__init__(self)

        if quality is not None and not is_valid_quality(quality):
            raise ValueError('Invalid JPEG quality: requires int in [1,100] or an iterable with least 2 such numbers')

        # Sanitize inputs
        if rounding_approximation is not None and rounding_approximation not in ['sin', 'harmonic', 'soft']:
            raise ValueError('Unsupported rounding approximation: {}'.format(rounding_approximation))

        # Quantization tables
        if trainable:
            q_mtx_luma_init = np.ones((8, 8), dtype=np.float32) if not is_number(quality) else jpeg_qtable(quality, 0)
            q_mtx_chroma_init = np.ones((8, 8), dtype=np.float32) if not is_number(quality) else jpeg_qtable(quality, 1)
            self._q_mtx_luma = self.add_weight('Q_mtx_luma', [8, 8], dtype=tf.float32, initializer=tf.constant_initializer(q_mtx_luma_init))
            self._q_mtx_chroma = self.add_weight('Q_mtx_chroma', [8, 8], dtype=tf.float32, initializer=tf.constant_initializer(q_mtx_chroma_init))
        else:
            self._q_mtx_luma = np.ones((8, 8), dtype=np.float32) if not is_number(quality) else jpeg_qtable(quality, 0)
            self._q_mtx_chroma = np.ones((8, 8), dtype=np.float32) if not is_number(quality) else jpeg_qtable(quality, 1)

        # Parameters
        self.quality = quality
        self.trainable = trainable
        self.rounding_approximation = rounding_approximation
        self.rounding_approximation_steps = rounding_approximation_steps

        # RGB to YCbCr conversion
        self._color_F = np.array([[0, 0.299, 0.587, 0.114], [128, -0.168736, -0.331264, 0.5], [128, 0.5, -0.418688, -0.081312]], dtype=np.float32)
        self._color_I = np.array([[-1.402 * 128, 1, 0, 1.402], [1.058272 * 128, 1, -0.344136, -0.714136], [-1.772 * 128, 1, 1.772, 0]], dtype=np.float32)
        
        # DCT
        self._dct_F = np.array([[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
                                [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                                [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                                [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
                                [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
                                [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
                                [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
                                [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]], dtype=np.float32)
        self._dct_I = np.transpose(self._dct_F)
        
        # Quantization layer
        self.quantization = Quantization(self.rounding_approximation, self.rounding_approximation_steps, latent_bpf=9)

    def call(self, inputs):
        block_size = 8

        with tf.name_scope('jpeg'):

            # Color conversion (RGB -> YCbCr)
            with tf.name_scope('rgb_to_ycbcr'):
                            
                xc = tf.pad(255.0 * inputs, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=1)
                ycbcr = tf.nn.conv2d(xc, tf.reshape(tf.transpose(self._color_F), [1, 1, 4, 3]), [1, 1, 1, 1], 'SAME')

            with tf.name_scope('blocking'):
                # Re-organize to get non-overlapping blocks in the following form
                # (n_examples * 3, block_size, block_size, n_blocks)
                p = tf.transpose(ycbcr - 127, [0, 3, 1, 2])
                p = tf.reshape(p, [-1, tf.shape(p)[2], tf.shape(p)[3]])
                p = tf.expand_dims(p, axis=3)
                p = tf.nn.space_to_depth(p, block_size)
                p = tf.transpose(p, [0, 3, 1, 2])
                p = tf.reshape(p, [-1, block_size, block_size, tf.shape(p)[2] * tf.shape(p)[3]])

                # Reorganize to move n_blocks to the first dimension
                r = tf.transpose(p, [0, 3, 1, 2])
                r = tf.reshape(r, [-1, r.shape[2], r.shape[3]])

            # Forward DCT transform
            with tf.name_scope('dct'):
                Xi = tf.matmul(tf.tile(tf.expand_dims(self._dct_F, axis=0), [tf.shape(r)[0], 1, 1]), r)
                X = tf.matmul(Xi, tf.tile(tf.expand_dims(self._dct_I, axis=0), [tf.shape(r)[0], 1, 1]))

            # Approximate quantization
            with tf.name_scope('quantization'):
                # Tile quantization values for successive channels: 
                # image_0 [R .. R G .. G B .. B] ... image_N [R .. R G .. G B .. B]
                Ql = tf.tile(tf.expand_dims(self._q_mtx_luma, axis=0), [1 * (tf.shape(p)[-1]), 1, 1])
                Qc = tf.tile(tf.expand_dims(self._q_mtx_chroma, axis=0), [2 * (tf.shape(p)[-1]), 1, 1])
                Q = tf.concat((Ql, Qc), axis=0)
                Q = tf.tile(Q, [(tf.shape(inputs)[0]), 1, 1])
                X = X / Q
                X = self.quantization(X)
                X = X * Q

            with tf.name_scope('idct'):
                # Inverse DCT transform
                xi = tf.matmul(tf.tile(tf.expand_dims(self._dct_I, axis=0), [tf.shape(r)[0], 1, 1]), X)
                xi = tf.matmul(xi, tf.tile(tf.expand_dims(self._dct_F, axis=0), [tf.shape(r)[0], 1, 1]))

            with tf.name_scope('rev-blocking'):
                # Reorganize data back to
                xi = tf.reshape(xi, [3 * tf.shape(inputs)[0], -1, xi.shape[1], xi.shape[2]])
                xi = tf.transpose(xi, [0, 2, 3, 1])

                # Backward re-organization from blocks
                # (n_examples * 3, block, block, n_blocks) -> (n_examples, w, h, 3)
                q = tf.reshape(xi, [-1, tf.shape(xi)[1] * tf.shape(xi)[2], tf.shape(inputs)[1] // block_size,
                                    tf.shape(inputs)[2] // block_size])
                q = tf.transpose(q, [0, 2, 3, 1])
                q = tf.nn.depth_to_space(q, block_size)
                q = tf.reshape(q, [-1, 3, tf.shape(q)[1], tf.shape(q)[2]])
                q = tf.transpose(q, [0, 2, 3, 1])

            # Color conversion (YCbCr-> RGB)
            with tf.name_scope('ycbcr_to_rgb'):
                qc = tf.pad(q + 127, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=1)
                y = tf.nn.conv2d(qc, tf.reshape(tf.transpose(self._color_I), [1, 1, 4, 3]), [1, 1, 1, 1], 'SAME')
                y = y / 255.0                    
                y = tf.clip_by_value(y, 0, 1)

        return y, X


class JPEG(TFModel):
    """
    Generic model that provides JPEG compression to the framework. It can use either:
      1. a differentiable approximation of the JPEG codec,
      2. the standard JPEG codec (libjpeg via imageio)
    Typically (1) is used for training, and (2) is used for final validation.

    The class also provides randomization of the compression quality - useful for data augmentation.

    For ad-hoc compression needs, consider using 'differentiable_jpeg()' which delegates to a single, 
    lazy initialized, instance of the codec.

    For use-cases outside of the framework, the 'DifferentiableJPEG' class (tf.keras.Model) may be more appropriate.
    """

    def __init__(self, quality=None, codec='soft', trainable=False):
        """
        :param quality: JPEG quality level or None (can be specified later)
        :param codec: 'libjpeg', 'soft', 'sin', 'harmonic'
        :param trainable: set true to make the quantization tables trainable (under development)
        """
        super().__init__()

        # Sanitize inputs
        if codec is not None and codec not in ['libjpeg', 'soft', 'sin', 'harmonic']:
            raise ValueError('Unsupported codec version: {}'.format(codec))

        if codec == 'libjpeg':
            self._model = None
        else:
            self._model = DifferentiableJPEG(quality, codec, trainable=trainable)

        # Remember settings
        self.codec = codec
        self.quality = quality
        self.loss =  tf.keras.losses.MeanSquaredError()

    def reset_performance_stats(self):
        self._reset_performance(['entropy', 'ssim', 'psnr'])

    def process(self, batch_x, quality=None, return_entropy=False):
        """ Compress a batch of images (NHW3:rgb) with a given quality factor:

        - if quality is a number - use this quality level
        - if quality is an iterable with 2 numbers - use a random integer from that range
        - if quality is an iterable with >2 numbers - use a random value from that set
        """

        quality = self.quality if quality is None else quality

        if not is_valid_quality(quality):
            raise ValueError('Invalid or unspecified JPEG quality!')

        if hasattr(quality, '__getitem__') and len(quality) > 2:
            quality = int(np.random.choice(quality))
        
        elif hasattr(quality, '__getitem__') and len(quality) == 2:
            quality = np.random.randint(quality[0], quality[1])
        
        elif is_number(quality) and quality >= 1 and quality <= 100:
            quality = int(quality)
        
        else:
            raise ValueError('Invalid quality! {}'.format(quality))

        if self._model is None:
            if not isinstance(batch_x, np.ndarray):
                batch_x = batch_x.numpy()
            if return_entropy:
                return jpeg_helpers.compress_batch(batch_x, quality)[0], np.nan
            else:
                return jpeg_helpers.compress_batch(batch_x, quality)[0]
        else:
            if quality != self.quality:
                old_q_luma, old_q_chroma = self._model._q_mtx_luma, self._model._q_mtx_chroma
                self._model._q_mtx_luma = jpeg_qtable(quality, 0)
                self._model._q_mtx_chroma = jpeg_qtable(quality, 1)
            
            y, X = self._model(batch_x)

            if quality != self.quality:
                self._model._q_mtx_luma, self._model._q_mtx_chroma = old_q_luma, old_q_chroma

            if return_entropy:
                # TODO This currently takes too much memory
                # entropy = tf_helpers.entropy(X, self._model.quantization.codebook)[0]
                entropy = np.nan
                return y, entropy

            return y

    def __repr__(self):
        if self._model is not None:
            return 'JPEG(quality={},codec="{}",trainable={})'.format(self.quality, self.codec, self._model.trainable)
        else:
            return 'JPEG(quality={},codec="{}")'.format(self.quality, self.codec)

    def summary(self, quality=None):
        return f'JPEG ({self.codec}) {self._quality_mode(quality)}'

    def summary_compact(self, quality=None):
        return f'JPEG ({self.codec}) {self._quality_mode(quality)}'

    def estimate_qf(self, channel=0):
        """
        Estimate current JPEG quality factor (smallest difference wrt IJG tables) using luma (channel=0) or chroma (1) tables.
        """
        return jpeg_qf_estimation(self._model._q_mtx_luma, channel)

    def _quality_mode(self, quality=None):
        """ Human-readable assessment of the current JPEG quality settings. """
        quality = quality or self.quality
        if self._model is not None and self._model.trainable:
            return 'trainable QF~{}/{}'.format(
                jpeg_qf_estimation(self._model._q_mtx_luma, 0),
                jpeg_qf_estimation(self._model._q_mtx_chroma, 1)
                )
        elif is_number(quality):
            return 'QF={}'.format(quality)
        elif hasattr(quality, '__getitem__') and len(quality) == 2:
            return 'QF~[{},{}]'.format(*quality)
        elif hasattr(quality, '__getitem__') and len(quality) > 2:
            return 'QF~{{{}}}'.format(','.join(str(x) for x in quality))
        else:
            return 'QF=?'