import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from models.tfmodel import TFModel
from helpers import utils, tf_helpers, paramspec


class TwitterDCN(TFModel):
    """
    Auto-encoder architecture described in:
    [1] L. Theis, W. Shi, A. Cunningham, and F. Huszár, “Lossy Image Compression with Compressive Autoencoders,” Mar. 2017.
    """
    
    def construct_model(self, params):

        # Define expected hyper parameters and their values
        self._h = paramspec.ParamSpec({
            'n_filters': (8, int, (2, None)),
            'n_fscale': (2.0, float, (0.25, 4)),
            'n_latent': (0, int, (-1, None)),
            'kernel': (5, int, {3, 5, 7, 9, 11}),
            'n_layers': (9, int, (1, np.log2(self.patch_size) if self.patch_size is not None else 10)),  # Ensure valid latent representation
            'res_layers': (0, int, (0, 3)),
            'rounding': ('soft', str, {'identity', 'soft', 'soft-codebook', 'sin'}),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })

        self._h.update(**params)

        activation = tf.nn.leaky_relu
        latent_activation = tf.nn.tanh
        last_activation = tf.nn.sigmoid
        
        self.n_layers = 9
        self.latent_shape = (1, self.patch_size // 8, self.patch_size // 8, 96)
        self.n_latent = int(np.prod(self.latent_shape))

        print('Building Deep Compression Network with d-latent={}'.format(self.n_latent))

        with tf.name_scope('{}/encoder/normalization'.format(self.scoped_name)):
            net = 2 * (self.x - 0.5)
            print('net size: {}'.format(net.shape))

        # Encoder ---------------------------------------------------------------------------------------------------------
        
        net = tf.contrib.layers.conv2d(net,  64, 5, stride=2, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.scoped_name, 0))
        net = tf.contrib.layers.conv2d(net, 128, 5, stride=2, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.scoped_name, 1))
        
        resnet = tf.contrib.layers.conv2d(tf.nn.leaky_relu(net, name='{}/encoder/conv_{}/lrelu'.format(self.scoped_name, 1)), 128, 3, stride=1, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.scoped_name, 2))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.scoped_name, 3))
        net = tf.add(net, resnet, name='{}/encoder/sum_a{}'.format(self.scoped_name, 0))

        resnet = tf.contrib.layers.conv2d(net, 128, 3, stride=1, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.scoped_name, 4))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.scoped_name, 5))
        net = tf.add(net, resnet, name='{}/encoder/sum_b{}'.format(self.scoped_name, 1))

        resnet = tf.contrib.layers.conv2d(net, 128, 3, stride=1, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.scoped_name, 6))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.scoped_name, 7))
        net = tf.add(net, resnet, name='{}/encoder/sum_c{}'.format(self.scoped_name, 2))
        
        net = tf.contrib.layers.conv2d(net, 96, 5, stride=2, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.scoped_name, 8))
        
        # Latent space -------------------------------------------------------------------------------------------------
        
        latent = tf.identity(net, name='{}/encoder/latent_raw'.format(self.model_name))

        # Use GDN to Gaussianize data
        if self.use_gdn:
            latent = tf.contrib.layers.GDN(latent)
            self.log('GDN: {}'.format(latent.shape))

        # Add batch norm to normalize the latent representation
        if self.use_batchnorm:
            self.pre_bn = latent  # TODO Temporarily added for debugging
            self.is_training = tf.placeholder(tf.bool, shape=(), name='{}/is_training'.format(self.scoped_name))
            latent = tf.contrib.layers.batch_norm(latent, scale=False, is_training=self.is_training,
                                                  scope='{}/encoder/bn_{}'.format(self.scoped_name, 0))
            self.log('batch norm: {}'.format(latent.shape))

        # Learn a scaling factor for the latent features to encourage greater values (facilitates quantization)
        if self.scale_latent:
            # scaling_factor = np.max((1, np.power(2, self.latent_bpf - 2)))
            scaling_factor = 1
            alphas = tf.get_variable('{}/encoder/latent_scaling'.format(self.scoped_name), shape=(), dtype=tf.float32,
                                     initializer=tf.constant_initializer(scaling_factor))
            latent = tf.multiply(alphas, latent, name='{}/encoder/latent_scaled'.format(self.scoped_name))
            self.log('scaling latent representation - init:{}'.format(scaling_factor))

        # Add identity to facilitate better display in the TF graph
        latent = tf.identity(latent, name='{}/latent'.format(self.scoped_name))
        self.n_latent = int(np.prod(latent.shape[1:]))

        # Quantize the latent representation and remember tensors before and after the process
        self.latent_pre = latent
        latent = tf_helpers.quantization(latent, '{}/quantization'.format(self.scoped_name), 'latent_quantized',
                                         self._h.rounding, codebook_tensor=self.codebook)
        self.log('quantization with {} rounding'.format(self._h.rounding))
        self.latent_post = latent
        self.log('latent size: {} + quant:{}'.format(latent.shape, self._h.rounding))

        # Decoder ---------------------------------------------------------------------------------------------------

        inet = tf.contrib.layers.conv2d(latent, 512, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.scoped_name, 0))
        inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.scoped_name, 0))
        
        resnet = tf.contrib.layers.conv2d(inet, 128, 3, stride=1, activation_fn=activation, scope='{}/decoder/conv_{}'.format(self.scoped_name, 1))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.scoped_name, 2))
        inet = tf.add(inet, resnet, name='{}/decoder/sum_a{}'.format(self.scoped_name, 0))
        
        resnet = tf.contrib.layers.conv2d(inet, 128, 3, stride=1, activation_fn=activation, scope='{}/decoder/conv_{}'.format(self.scoped_name, 3))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.scoped_name, 4))
        inet = tf.add(inet, resnet, name='{}/decoder/sum_b{}'.format(self.scoped_name, 1))
        
        resnet = tf.contrib.layers.conv2d(inet, 128, 3, stride=1, activation_fn=activation, scope='{}/decoder/conv_{}'.format(self.scoped_name, 5))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.scoped_name, 6))
        inet = tf.add(inet, resnet, name='{}/decoder/sum_c{}'.format(self.scoped_name, 2))

        inet = tf.contrib.layers.conv2d(inet, 256, 3, stride=1, activation_fn=activation, scope='{}/decoder/tconv_{}'.format(self.scoped_name, 7))
        inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.scoped_name, 7))
        
        inet = tf.contrib.layers.conv2d(inet, 12, 3, stride=1, activation_fn=None, scope='{}/decoder/tconv_{}'.format(self.scoped_name, 8))
        inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.scoped_name, 8))
        
        with tf.name_scope('{}/decoder/denormalization'.format(self.scoped_name)):
            y = (inet + 1) / 2
            
        y = tf.identity(y, name="y")
        
        self.y = y
        self.latent = latent