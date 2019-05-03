import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from models.tfmodel import TFModel
from helpers import utils, tf_helpers


class TwitterDCN(TFModel):
    """
    Auto-encoder architecture described in:
    [1] L. Theis, W. Shi, A. Cunningham, and F. Huszár, “Lossy Image Compression with Compressive Autoencoders,” Mar. 2017.
    """
    
    def construct_model(self, train_codebook=False, latent_bpf=6, scale_latent=True, entropy_weight=None):
        
        activation = tf.nn.leaky_relu
        latent_activation = tf.nn.tanh
        last_activation = tf.nn.sigmoid
        
        self.n_layers = 9
        self.latent_shape = (1, self.patch_size // 8, self.patch_size // 8, 96)
        self.n_latent = int(np.prod(self.latent_shape))
        self.train_codebook = train_codebook
        self.latent_bpf = latent_bpf
        self.scale_latent = scale_latent
        self.entropy_weight = entropy_weight

        print('Building Deep Compression Network with d-latent={}'.format(self.n_latent))
        
        
        with tf.name_scope('{}/encoder/normalization'.format(self.model_name)):
            net = 2 * (self.x - 0.5)
            print('net size: {}'.format(net.shape))

        # Encoder ---------------------------------------------------------------------------------------------------------
        
        net = tf.contrib.layers.conv2d(net,  64, 5, stride=2, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.model_name, 0))
        net = tf.contrib.layers.conv2d(net, 128, 5, stride=2, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.model_name, 1))
        
        resnet = tf.contrib.layers.conv2d(tf.nn.leaky_relu(net, name='{}/encoder/conv_{}/lrelu'.format(self.model_name, 1)), 128, 3, stride=1, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.model_name, 2))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.model_name, 3))
        net = tf.add(net, resnet, name='{}/encoder/sum_a{}'.format(self.model_name, 0))

        resnet = tf.contrib.layers.conv2d(net, 128, 3, stride=1, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.model_name, 4))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.model_name, 5))
        net = tf.add(net, resnet, name='{}/encoder/sum_b{}'.format(self.model_name, 1))

        resnet = tf.contrib.layers.conv2d(net, 128, 3, stride=1, activation_fn=activation, scope='{}/encoder/conv_{}'.format(self.model_name, 6))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.model_name, 7))
        net = tf.add(net, resnet, name='{}/encoder/sum_c{}'.format(self.model_name, 2))
        
        net = tf.contrib.layers.conv2d(net, 96, 5, stride=2, activation_fn=None, scope='{}/encoder/conv_{}'.format(self.model_name, 8))
        
        # Latent space ----------------------------------------------------------------------------------------------------
        
        latent = tf.identity(net, name='{}/encoder/latent_raw'.format(self.model_name))
        
        # Learn a scaling factor for the latent features to encourage greater values (facilitates quantization)
        if self.scale_latent:
            scaling_factor = np.max((1, np.power(2, self.latent_bpf - 2)))
            alphas = tf.get_variable('{}/encoder/latent_scaling'.format(self.model_name), shape=(), dtype=tf.float32, initializer=tf.constant_initializer(scaling_factor))
            latent = tf.multiply(alphas, latent, name='{}/encoder/latent_scaled'.format(self.model_name))            
            print('Scaling latent representation - init:{}'.format(scaling_factor))
        
        # Add identity to facilitate better display in the TF graph
        latent = tf.identity(latent, name='{}/latent'.format(self.model_name))

        # Quantize the latent representation and remember tensors before and after the process
        self.latent_pre = latent
        latent = tf_helpers.quantization(latent, '{}/quantization'.format(self.model_name), 'latent_quantized', rounding)                        
        print('quantization with {} rounding'.format(rounding))
        self.latent_post = latent
        print('latent size: {} + quant:{}'.format(latent.shape, rounding))

        # Decoder ----------------------------------------------------------------------------------------------------------
        
                
        inet = tf.contrib.layers.conv2d(latent, 512, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.model_name, 0))
        inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.model_name, 0))
        
        resnet = tf.contrib.layers.conv2d(inet, 128, 3, stride=1, activation_fn=activation, scope='{}/decoder/conv_{}'.format(self.model_name, 1))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.model_name, 2))
        inet = tf.add(inet, resnet, name='{}/decoder/sum_a{}'.format(self.model_name, 0))
        
        resnet = tf.contrib.layers.conv2d(inet, 128, 3, stride=1, activation_fn=activation, scope='{}/decoder/conv_{}'.format(self.model_name, 3))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.model_name, 4))
        inet = tf.add(inet, resnet, name='{}/decoder/sum_b{}'.format(self.model_name, 1))
        
        resnet = tf.contrib.layers.conv2d(inet, 128, 3, stride=1, activation_fn=activation, scope='{}/decoder/conv_{}'.format(self.model_name, 5))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, stride=1, activation_fn=None, scope='{}/decoder/conv_{}'.format(self.model_name, 6))
        inet = tf.add(inet, resnet, name='{}/decoder/sum_c{}'.format(self.model_name, 2))

        inet = tf.contrib.layers.conv2d(inet, 256, 3, stride=1, activation_fn=activation, scope='{}/decoder/tconv_{}'.format(self.model_name, 7))
        inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.model_name, 7))
        
        inet = tf.contrib.layers.conv2d(inet, 12, 3, stride=1, activation_fn=None, scope='{}/decoder/tconv_{}'.format(self.model_name, 8))
        inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.model_name, 8))
        
        with tf.name_scope('{}/decoder/denormalization'.format(self.model_name)):
            y = (inet + 1) / 2
            
        y = tf.identity(y, name="y")
        
        self.y = y
        self.latent = latent