# -*- coding: utf-8 -*-
"""
Custom TF layers for reuse in other models.
"""
import numpy as np
import tensorflow as tf

import helpers.kernels
from helpers import tf_helpers


class ConstrainedConv2D(tf.keras.layers.Layer):
    """
    Implementation of a trainable constrained residual filter (based on [1] and extended to RGB inputs). 
    The layer learns a 2D convolution filter (5, 5, 3, 3) where:

    - a central pixel in each channel [2, 2, i, i] is set to a fixed negative value
    - each output channel is normalized to sum to 0 [:, :, :, i]
    
    For example, an intra-channel filter [:, :, i, i] may look like:

    [  -0.73    0.41   -1.24   -1.26    0.69]
    [  -0.29    7.91   17.53    8.6     0.29]
    [  -0.62   16.7  -100.0    16.1     0.22]
    [   0.54    9.3    16.05    8.19    0.98]
    [  -0.57   -0.7    -0.4     1.22   -0.13]

    The layer is pre-initialized with a simple residual filter with no intra-channel interactions.

    # References

    [1] Bayar & Stamm, Constrained convolutional neural networks: A new approach towards general purpose image 
        manipulation detection. IEEE Transactions on Information Forensics and Security, 13 (11), 2018
    """

    def __init__(self, filter_strength=100, trainable=True):
        super().__init__()
        self.filter_strength = filter_strength

        f = np.array([[0, 0, 0, 0, 0], [0, -1, -2, -1, 0], [0, -2, 12, -2, 0], [0, -1, -2, -1, 0], [0, 0, 0, 0, 0]])
        self.kernel = self.add_weight("kernel", shape=(5, 5, 3, 3),
                                      initializer=tf.constant_initializer(helpers.kernels.repeat_2dfilter(f, 3)),
                                      trainable=trainable)

    def call(self, input):
        # Mask for normalizing the residual filter
        tf_ind = tf.constant(helpers.kernels.center_mask_2dfilter(5, 3), dtype=tf.float32)

        # Normalize the residual filter
        nf = self.kernel * (1 - tf_ind)
        df = tf.tile(tf.reshape(tf.reduce_sum(nf, axis=(0,1,2)), [1, 1, 1, 3]), [5, 5, 3, 1])
        nf = self.filter_strength * nf / df
        nf = nf - self.filter_strength * tf_ind

        # Convolution with the residual filter
        xp = tf.pad(input, [[0, 0], [2, 2], [2, 2], [0, 0]], 'SYMMETRIC')
        return tf.nn.conv2d(xp, nf, [1, 1, 1, 1], 'VALID')


class Quantization(tf.keras.layers.Layer):
    """
    A (differentiable) quantization layer. Supported quantization modes:

    - round: simple rounding; NOT differentiable
    - sin: sinusoidal approximation to rounding both during forward and backward pass; differentiable
    - soft: simple rounding in the forward pass, sinusoidal approx. in the backward pass; differentiable
    - identity: identity function, potentially useful when debugging  
    - harmonic: harmonic approximation based on Taylor expansion up to 'taylor_terms' terms; differentiable
    - soft-codebook: soft approximation based on distance to quantization code-book entires; differentiable

    The quantization codebook is set-up based on the desired number of bits per feature ('latent_bpf'), e.g.:
    - 3 bpf will yield a codebook with 8=2^3 entries -> [-3, -2, -1, 0, 1, 2, 3, 4]

    The distance w.r.t. code-book entires can be computed with a Gaussian or t-Student kernel. 

    """

    def __init__(self, rounding='soft', v=50, gamma=25, latent_bpf=4, trainable=False, taylor_terms=1):
        """
        Note that not all parameters are applicable to all approximation modes.

        :param rounding: method of rounding approximation

        # Soft-codebook approximation

        :param v: degrees of freedom for the distance kernel (0 -> Gaussian, >0 -> t-Student)
        :param gamma: controls the scale of the kernel
        :param latent_bpf: range of the of the quantization codebook - specified in bits/feature
        :param trainable: option to make the codebook trainable (not tested)

        # Harmonic approximation

        :param taylor_terms: number of terms for the harmonic Taylor approximation
        """
        super().__init__()

        if rounding not in {'round', 'sin', 'soft', 'identity', 'harmonic', 'soft-codebook'}:
            raise ValueError('Unsupported quantization: {}'.format(rounding))

        self.rounding = rounding
        self.taylor_terms = taylor_terms
        self.v = v
        self.gamma = gamma
        self.latent_bpf = latent_bpf
        self.trainable = trainable

        # Setup codebook
        # Even if the codebook is not used for quantization, it may be used for entropy estimation somewhere else
        # TODO Seemingly unnecessary codebook init (should this be moved/fixed?)
        qmin = -2 ** (self.latent_bpf - 1) + 1
        qmax = 2 ** (self.latent_bpf - 1)
                            
        if self.trainable:
            self.codebook = self.add_weight(initializer=tf.constant_initializer(np.arange(qmin, qmax + 1)), shape=(1, 2 ** self.latent_bpf), dtype=tf.float32)
        else:
            self.codebook = tf.constant(np.arange(qmin, qmax + 1), shape=(1, 2 ** self.latent_bpf), dtype=tf.float32)

    def call(self, x):

        if self.rounding == 'round':
            x = tf.round(x)

        elif self.rounding == 'sin':
            x = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi))

        elif self.rounding == 'soft':
            x_ = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi))
            x = tf.add(tf.stop_gradient(tf.round(x) - x_), x_)

        elif self.rounding == 'harmonic':
            xa = x - tf.sin(2 * np.pi * x) / np.pi
            for k in range(2, self.taylor_terms):
                xa += tf.pow(-1.0, k) * tf.sin(2 * np.pi * k * x) / (k * np.pi)
            x = xa

        elif self.rounding == 'identity':
            x = x

        elif self.rounding == 'soft-codebook':

            prec_dtype = tf.float64
            eps = 1e-72

            assert(self.codebook.shape[0] == 1)
            assert(self.codebook.shape[1] > 1)

            values = tf.reshape(x, (-1, 1))

            if self.v <= 0:
                # Gaussian soft quantization
                weights = tf.exp(-self.gamma * tf.pow(tf.cast(values, dtype=prec_dtype) - tf.cast(self.codebook, dtype=prec_dtype), 2))
            else:
                # t-Student soft quantization
                dff = tf.cast(values, dtype=prec_dtype) - tf.cast(self.codebook, dtype=prec_dtype)
                dff = self.gamma * dff
                weights = tf.pow((1 + tf.pow(dff, 2)/self.v), -(self.v+1)/2)

            weights = (weights + eps) / (tf.reduce_sum(weights + eps, axis=1, keepdims=True))

            assert(weights.shape[1] == np.prod(self.codebook.shape))

            soft = tf.reduce_mean(tf.matmul(weights, tf.transpose(tf.cast(self.codebook, dtype=prec_dtype))), axis=1)
            soft = tf.cast(soft, dtype=tf.float32)
            soft = tf.reshape(soft, tf.shape(x))

            hard = tf.gather(self.codebook, tf.argmax(weights, axis=1), axis=1)
            hard = tf.reshape(hard, tf.shape(x))

            x = tf.stop_gradient(hard - soft) + soft
            x = tf.identity(x)

        return x


class DiscreteLatent(tf.keras.layers.Layer):
    """
    A quantization layer with additional control over representation entropy. The layer adds a trainable scaling factor
    which can help increase activation range to match the quantization codebook.

    See also: 'Quantization' layer
    """

    def __init__(self, rounding='soft', v=50, gamma=25, latent_bpf=4, trainable_codebook=False, trainable_scale=True):
        super(DiscreteLatent, self).__init__()
        self.trainable_scale = trainable_scale
        self.rounding = rounding
        self.v = v
        self.gamma = gamma
        self.latent_bpf = latent_bpf
        self.trainable_codebook = trainable_codebook
        if self.trainable_scale:
            self.scaling_factor = self.add_weight(shape=(), dtype=tf.float32, initializer=tf.constant_initializer(1), name='latent_scaling')
        self.quantization = Quantization(rounding, v, gamma, latent_bpf, trainable_codebook)

    def call(self, inputs):
        latent = inputs
        if self.trainable_scale:
            latent = latent * self.scaling_factor

        latent = self.quantization(latent)
        entropy_ = tf_helpers.entropy(latent, self.quantization.codebook, self.v, self.gamma)[0]

        return latent, entropy_


class DemosaicingLayer(tf.keras.layers.Layer):
        
    def __init__(self, c_filters, kernel, activation, residual, **kwargs):
        """
        :param c_filters: a tuple with the numbers of filters for initial conv layers
        :param io_filters: the number of filters in the final 1x1 convolution
        :param kernel: kernel size for the initial convolutions
        :param activation: activation function (string, see tf_helpers.activation_mapping)
        """
        super().__init__(**kwargs)
        activation = tf_helpers.activation_mapping[activation]
        if residual:
            self._bilinear_kernel = helpers.kernels.bilin_kernel(kernel)
            self._pad = (kernel - 1) // 2
            self._bilinear = tf.keras.layers.Conv2D(3, kernel, kernel_initializer=tf.constant_initializer(self._bilinear_kernel), use_bias=False, activation=None, padding='VALID', trainable=False)
            self._alpha = self.add_weight("alpha", initializer=tf.constant_initializer(0.1))
        else:
            self._bilinear = None
        
        self._layers = []

        # Setup conv layers
        for n_filters in c_filters:
            self._layers.append(tf.keras.layers.Conv2D(n_filters, kernel, 1, 'same', activation=activation))

        # Final 1x1 conv to project all features to the RGB color space
        self._layers.append(tf.keras.layers.Conv2D(3, 1, 1, 'same', 
            activation=tf.keras.activations.tanh if residual else tf.keras.activations.sigmoid))
        
    def call(self, inputs, clip=True):
        # Learn the RGB output directly
        if self._bilinear is None:
            f = inputs
            for l in self._layers:
                f = l(f)
            y = f
        
        # Learn a residual wrt a bilinear filter
        else:
            bayer = tf.pad(inputs, tf.constant([[0, 0], [self._pad, self._pad], [self._pad, self._pad], [0, 0]]), 'REFLECT')
            x = self._bilinear(bayer)
            if len(self._layers) > 1:
                f = inputs
                for l in self._layers:
                    f = l(f)
            else:
                f = 0
            y = x - self._alpha * f

        if clip:
            y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        
        return y
