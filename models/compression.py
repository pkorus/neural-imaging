import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

from models.tfmodel import TFModel
from helpers import utils, tf_helpers


class DCN(TFModel):
    """
    A forensic analysis network with the following architecture:

    1. A constrained conv layer (learned residual filter)
    2. N x standard conv layers
    3. A 1x1 conv layer
    4. GAP for feature extraction
    5. 2 hidden fully connected layers
    6. Output layer with K classes
    """

    def __init__(self, sess, graph, label=None, x=None, nip_input=None, patch_size=128, latent_bpf=4, train_codebook=False, entropy_weight=None, default_val_is_train=True, scale_latent=False, use_batchnorm=False, **kwargs):
        """
        Creates a forensic analysis network.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        """
        super().__init__(sess, graph, label)
        self.patch_size = patch_size
        self.nip_input = nip_input
        self.latent_bpf = latent_bpf
        self.train_codebook = train_codebook
        self.entropy_weight = entropy_weight
        self.default_val_is_train = default_val_is_train
        self.scale_latent = scale_latent
        self.use_batchnorm = use_batchnorm

        # Remember parameters passed to the constructor
        self.args = kwargs

        with self.graph.as_default():
            
            # Setup inputs:
            # - if possible take external tensor as input, otherwise create a placeholder
            # - if external input is given (from a NIP model), remember the input to the NIP model to facilitate 
            #   convenient operation of the class (see helper methods 'process*')
            if x is None:
                x = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 3), name='x_{}'.format(self.scoped_name))
                self.use_nip_input = False
            else:
                self.use_nip_input = True
            
            self.x = x
            
            # Setup quantization codebook
            with tf.name_scope('{}/optimization'.format(self.scoped_name)):
                
                with tf.name_scope('entropy'):
                                        
                    # Initialize the quantization codebook
                    qmin = -2 ** (self.latent_bpf - 1) + 1
                    qmax = 2 ** (self.latent_bpf - 1)
                                        
                    print('Initializing {} codebook ({} bpf): from {} to {}'.format('trainable' if self.train_codebook else 'fixed', self.latent_bpf, qmin, qmax))
                    if self.train_codebook:
                        bin_centers = tf.get_variable('{}/quantization/codebook'.format(self.scoped_name), shape=(1, 2 ** self.latent_bpf), initializer=tf.constant_initializer(np.arange(qmin, qmax + 1)))
                    else:
                        bin_centers = tf.constant(np.arange(qmin, qmax + 1), shape=(1, 2 ** self.latent_bpf), dtype=tf.float32)                        
                    self.codebook = bin_centers                
            
            # Construct the actual model
            self.construct_model(kwargs)

            # Check if the model has set all expected attributes
            setup_status = {key: hasattr(self, key) for key in ['y', 'latent_pre', 'latent_post', 'latent_shape', 'n_latent', 'train_codebook', 'latent_bpf', 'entropy_weight']}
            if not all(setup_status.values()):
                raise NotImplementedError('The model construction function has failed to set-up some attributes: {}'.format([key for key, value in setup_status.items() if not value]))
                
            # train_codebook=False, latent_bpf=8, scale_latent=True, entropy_weight=None
                        
            with tf.name_scope('{}/optimization'.format(self.scoped_name)):
                
                with tf.name_scope('entropy'):

                    # Estimate entropy of the latent representation
                    
                    # Some parameters
                    soft_quantization_sigma = 5
                    prec_dtype = tf.float64
                    v = 100
                    eps = 1e-72

                    values = tf.reshape(self.latent_pre, (-1, 1))
                    
                    assert(self.codebook.shape[0] == 1)
                    assert(self.codebook.shape[1] > 1)                    
                    
                    # Compute soft-quantization
                    if v <= 0:
                        print('Entropy estimation using Gaussian soft quantization')
                        weights = tf.exp(-soft_quantization_sigma * tf.pow(values - self.codebook, 2)) 
                    else:
                        # t-Student-like distance measure with heavy tails
                        print('Entropy estimation using t-Student soft quantization')
                        dff = values - self.codebook
                        dff = soft_quantization_sigma * dff
                        weights = tf.pow((1 + tf.pow(dff, 2)/v), -(v+1)/2)

                    weights = (weights + eps) / (eps + tf.reduce_sum(weights, axis=1, keepdims=True))
                    
                    assert(weights.shape[1] == np.prod(self.codebook.shape))
                    
                    # Compute soft histogram
                    histogram = tf.reduce_mean(weights, axis=0)
#                     histogram = tf.clip_by_value(histogram, 1e-9, tf.float32.max)
                    histogram = tf.maximum(histogram, 1e-9)
                    histogram = histogram / tf.reduce_sum(histogram)

                    self.entropy = - tf.reduce_sum(histogram * tf.log(histogram) / 0.6931) # 0.6931 - log(2)
                    self.histogram = histogram
                    self.weights = weights
                
                # Loss and SSIM
                self.ssim = tf.reduce_mean(tf.image.ssim(self.x, tf.clip_by_value(self.y, 0, 1), max_val=1))
                self.loss = tf.nn.l2_loss(self.x - self.y)
                if self.entropy_weight is not None:
                    self.loss = self.loss + self.entropy_weight * self.entropy
                print('Initializing loss: L2 {}'.format('+ {} * entropy'.format(self.entropy_weight) if self.entropy_weight is not None else ''))
                
                # Optimization
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.lr = tf.placeholder(tf.float32, name='{}_learning_rate'.format(self.scoped_name))
                    self.adam = tf.train.AdamOptimizer(learning_rate=self.lr)
                    self.opt = self.adam.minimize(self.loss, var_list=self.parameters)
                
    def construct_model(self):
        raise NotImplementedError('Not implemented!')
        
    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'entropy': {'training': [], 'validation': []},
            'ssim': {'training': [], 'validation': []}
        }

    def compress(self, batch_x, is_training=None):
        with self.graph.as_default():
            
            feed_dict = {
                self.x if not self.use_nip_input else self.nip_input: batch_x,
            }            
            
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training or self.default_val_is_train
                
            y = self.sess.run(self.latent_post, feed_dict=feed_dict)
            return y

    def compress_soft(self, batch_x, is_training=None):
        with self.graph.as_default():
            
            feed_dict = {
                self.x if not self.use_nip_input else self.nip_input: batch_x,
            }
            
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training or self.default_val_is_train
            
            y = self.sess.run(self.latent_pre, feed_dict=feed_dict)
            return y        
        
    def decompress(self, batch_z):
        with self.graph.as_default():
            
            feed_dict = {
                self.latent_post: batch_z
            }
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = 1.0
                
            y = self.sess.run(self.y, feed_dict)
            return y.clip(0, 1)
            
    def process(self, batch_x, dropout_keep_prob=1.0, is_training=None):
        """
        Returns the predicted class for an image batch. The input is fed to the NIP if the model is chained properly.
        """
        with self.graph.as_default():
            
            feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x
            }
            
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = dropout_keep_prob
                
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training or self.default_val_is_train
              
            y = self.sess.run(self.y, feed_dict)
            return y.clip(0, 1)
    
    def process_direct(self, batch_x, dropout_keep_prob=1.0, is_training=None):
        """
        Returns the predicted class for an image batch. The input is always fed to the FAN model directly.
        """
        with self.graph.as_default():
            feed_dict = {
                self.x: batch_x
            }
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = dropout_keep_prob
                
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training or self.default_val_is_train
                
            y = self.sess.run(self.y, feed_dict)
            return y.clip(0, 1)
    
    def training_step(self, batch_x, learning_rate, dropout_keep_prob=1.0):
        """
        Make a single training step and return current loss. Only the FAN model is updated.
        """
        with self.graph.as_default():
            feed_dict = {
                    self.x if not self.use_nip_input else self.nip_input: batch_x,
                    self.lr: learning_rate
            }
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = dropout_keep_prob
                
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = True                
            
            _, loss, ssim, entropy = self.sess.run([self.opt, self.loss, self.ssim, self.entropy], feed_dict)
            return {
                'loss': np.sqrt(2 * loss), # The L2 loss in TF is computed differently (half of non-square rooted norm)
                'ssim': ssim,
                'entropy': entropy
            }

    def compression_stats(self, patch_size=None, n_latent_bytes=1):
        ps = patch_size or self.patch_size        
        if ps is None:
            raise ValueError('Patch size not specified!')
            
        bitmap_size = ps * ps * 3
        return {
            'rate': bitmap_size / (n_latent_bytes * self.n_latent),
            'bpp': 8 * self.n_latent * n_latent_bytes / (ps * ps),
            'bytes': self.n_latent * n_latent_bytes
        }
    
    def summary(self):
        return 'dcn with {} conv layers and {}-D latent representation [{:,} parameters]'.format(self.n_layers, self.n_latent, self.count_parameters())
    
    @property
    def model_code(self):
        if not hasattr(self, 'n_latent'):
            raise ValueError('The model does not report the latent space dimensionality.')
        
        return '{}-{}D'.format(type(self).__name__, self.n_latent)        
    
    
class AutoencoderDCN(DCN):

    def get_hyperparameters_specs(self):
        # Tuple with default value, dtype
        return {
            'n_filters': (8, int),
            'n_fscale': (2.0, float),
            'n_latent': (0, int),
            'kernel': (5, int),
            'n_layers': (3, int),
            'res_layers': (0, int),
            'dropout': (True, bool),
            'rounding': ('soft', str),
            # 'use_batchnorm': (True, bool),
            'train_codebook': (False, bool),
            # 'scale_latent': (True, bool),
            'activation': (tf.nn.leaky_relu, None)
        }

    def parse_hyperparameters(self, params):
        specs = self.get_hyperparameters_specs()

        # Apply defaults
        parameters = {key: spec[0] for key, spec in specs.items()}

        # Iterate over submitted values
        for key, value in params.items():

            if key in specs:
                # Get specification for the current parameter
                _, dtype = specs[key]

                # Accept the new value if it:
                #   is not None
                #   is not np.nan (for numerical types)
                if value is not None and (type(value) in [int, float, bool, np.int, np.float, np.bool] and not np.isnan(value)):
                    parameters[key] = value if dtype is None else dtype(value)

            else:
                raise ValueError('Unexpected parameter: {}!'.format(key))

        return parameters

    def construct_model(self, params):

        self.hyperparameters = self.parse_hyperparameters(params)

        # Extract parameters for easy access
        n_filters = self.hyperparameters['n_filters']
        n_fscale = self.hyperparameters['n_fscale']
        n_latent = self.hyperparameters['n_latent']
        n_layers = self.hyperparameters['n_layers']
        res_layers = self.hyperparameters['res_layers']
        kernel = self.hyperparameters['kernel']
        activation = self.hyperparameters['activation']
        rounding = self.hyperparameters['rounding']
        # use_batchnorm = self.hyperparameters['use_batchnorm']
        dropout = self.hyperparameters['dropout']
        # scale_latent = self.hyperparameters['scale_latent']

        # if n_layers < 1:
        #     raise ValueError('n_layers needs to be > 0!')

        self.uses_bottleneck = n_latent > 0

        latent_activation = None
        last_activation = None

        print('Building Deep Compression Network')

        net = self.x
        print('in size: {}'.format(net.shape))

        # Encoder ------------------------------------------------------------------------------------------------------

        # Add convolutional layers
        current_n_filters = n_filters

        for r in range(n_layers):
            current_activation = activation if (n_latent > 0 or (n_latent == 0 and r < n_layers - 1)) else latent_activation
            net = tf.contrib.layers.conv2d(net, current_n_filters, kernel, stride=2, scope='{}/encoder/conv_{}'.format(self.scoped_name, r), activation_fn=current_activation)
            print('conv size: {} + {}'.format(net.shape, current_activation.__name__ if current_activation is not None else None))
            if r != n_layers - 1:
                current_n_filters *= n_fscale

        # Add residual blocks
        for r in range(res_layers):
            resnet = tf.contrib.layers.conv2d(tf.nn.leaky_relu(net, name='{}/encoder/res_{}/lrelu'.format(self.scoped_name, r)), current_n_filters, 3, stride=1, activation_fn=activation, scope='{}/encoder/res_{}/conv_{}'.format(self.scoped_name, r, 0))
            resnet = tf.contrib.layers.conv2d(resnet, current_n_filters, 3, stride=1, activation_fn=None, scope='{}/encoder/res_{}/conv_{}'.format(self.scoped_name, r, 1))
            net = tf.add(net, resnet, name='{}/encoder/res_{}/sum'.format(self.scoped_name, r))
            print('residual block: {}'.format(net.shape))

            # Latent representation ----------------------------------------------------------------------------------------

        # Compute the shape of the latent representation
        z_spatial = int(self.patch_size / (2**n_layers))
        z_features = int(n_filters * (n_fscale**(n_layers-1)))
        self.latent_shape = [-1, z_spatial, z_spatial, z_features]

        # If a smaller linear bottleneck is specified explicitly - add dense layers to make the projection
        if n_latent is not None and n_latent != 0:
            flat = tf.contrib.layers.flatten(net, scope='{}/encoder/flatten_{}'.format(self.scoped_name, 0))
            print('flatten size: {}'.format(flat.shape))

            if n_latent > 0:
                flat = tf.contrib.layers.fully_connected(flat, n_latent, activation_fn=latent_activation, scope='{}/encoder/dense_{}'.format(self.scoped_name, 0))
                latent = tf.identity(flat, name='{}/encoder/latent_raw'.format(self.scoped_name))
                print('dense size: {}'.format(flat.shape))
            else:
                latent = tf.identity(flat, name='{}/encoder/latent_raw'.format(self.scoped_name))
        else:
            latent = tf.identity(net, name='{}/encoder/latent_raw'.format(self.scoped_name))

        # Add batch norm to normalize the latent representation
        if self.use_batchnorm:
            self.pre_bn = latent # TODO Temporarily added for debugging
            self.is_training = tf.placeholder(tf.bool, shape=(), name='{}/is_training'.format(self.scoped_name))
            #                 self.is_training = tf.get_variable('is_training', shape=(), dtype=tf.bool, initializer=tf.constant_initializer(True), trainable=False)

            latent = tf.contrib.layers.batch_norm(latent, scale=False, is_training=self.is_training, scope='{}/encoder/bn_{}'.format(self.scoped_name, 0))
            print('batch norm: {}'.format(latent.shape))

        # Learn a scaling factor for the latent features to encourage greater values (facilitates quantization)
        if self.scale_latent:
            #             scaling_factor = np.max((1, np.power(2, self.latent_bpf - 2)))
            scaling_factor = 1
            alphas = tf.get_variable('{}/encoder/latent_scaling'.format(self.scoped_name), shape=(), dtype=tf.float32, initializer=tf.constant_initializer(scaling_factor))
            latent = tf.multiply(alphas, latent, name='{}/encoder/latent_scaled'.format(self.scoped_name))
            print('Scaling latent representation - init:{}'.format(scaling_factor))

        # Add identity to facilitate better display in the TF graph
        latent = tf.identity(latent, name='{}/latent'.format(self.scoped_name))

        # Quantize the latent representation and remember tensors before and after the process
        self.latent_pre = latent
        latent = tf_helpers.quantization(latent, '{}/quantization'.format(self.scoped_name), 'latent_quantized', rounding, codebook_tensor=self.codebook)
        print('quantization with {} rounding'.format(rounding))
        self.latent_post = latent
        self.n_latent = int(np.prod(latent.shape[1:]))
        print('latent size: {} + quant:{}'.format(latent.shape, rounding))

        if n_latent > 0:
            inet = tf.contrib.layers.fully_connected(latent, int(np.prod(self.latent_shape[1:])), activation_fn=activation, scope='{}/decoder/dense_{}'.format(self.scoped_name, 0))
            print('dense size: {} + {}'.format(inet.shape, activation))
        else:
            inet = latent

        # Add dropout
        if dropout:
            if not hasattr(self, 'is_training'):
                self.is_training = tf.placeholder(tf.bool, shape=(), name='{}/is_training'.format(self.scoped_name))
            self.dropout = tf.placeholder(tf.float32, name='{}/droprate'.format(self.scoped_name), shape=())
            inet = tf.contrib.layers.dropout(inet, keep_prob=self.dropout, scope='{}/dropout'.format(self.scoped_name), is_training=self.is_training)
            print('dropout size: {}'.format(net.shape))

        # Decoder ------------------------------------------------------------------------------------------------------

        # Just in case - make sure we have a multidimensional tensor before we start the convolutions
        inet = tf.reshape(inet, self.latent_shape, name='{}/decoder/reshape_{}'.format(self.scoped_name, 0))
        print('reshape size: {}'.format(inet.shape))

        # Add residual blocks
        for r in range(res_layers):
            resnet = tf.contrib.layers.conv2d(tf.nn.leaky_relu(inet, name='{}/encoder/res_{}/lrelu'.format(self.scoped_name, r)), current_n_filters, 3, stride=1, activation_fn=activation, scope='{}/decoder/res_{}/conv_{}'.format(self.scoped_name, r, 0))
            resnet = tf.contrib.layers.conv2d(resnet, current_n_filters, 3, stride=1, activation_fn=None, scope='{}/decoder/res_{}/conv_{}'.format(self.scoped_name, r, 1))
            inet = tf.add(inet, resnet, name='{}/decoder/res_{}/sum'.format(self.scoped_name, r))
            print('residual block: {}'.format(net.shape))

            # Transposed convolutions
        for r in range(n_layers):
            current_activation = last_activation if r == n_layers - 1 else activation
            inet = tf.contrib.layers.conv2d(inet, 2 * current_n_filters, kernel, stride=1, scope='{}/decoder/tconv_{}'.format(self.scoped_name, r), activation_fn=current_activation)
            print('conv size: {} + {}'.format(inet.shape, current_activation.__name__ if current_activation is not None else None))
            inet = tf.depth_to_space(inet, 2, name='{}/decoder/d2s_{}'.format(self.scoped_name, r))
            #             inet = tf.contrib.layers.conv2d_transpose(inet, 3 if r == self.n_layers - 1 else n_filters, self.kernel, stride=2,  activation_fn=current_activation, scope='{}/tconv_{}'.format(self.scoped_name, r))
            print('d2s size: {} + {}'.format(inet.shape, None))
            current_n_filters = current_n_filters // n_fscale

        inet = tf.contrib.layers.conv2d(inet, 3, kernel, stride=1, activation_fn=last_activation, scope='{}/decoder/tconv_out'.format(self.scoped_name))
        print('conv->out size: {} + {}'.format(inet.shape, last_activation))
        y = tf.identity(inet, name='y')

        self.y = y
        self.latent = latent

    @property
    def model_code(self):
        parameter_summary = []

        if hasattr(self, 'latent_shape'):
            parameter_summary.append('x'.join(str(x) for x in self.latent_shape[1:]))

        layer_summary = []
        if 'n_layers' in self.hyperparameters:
            layer_summary.append('{:d}C'.format(int(self.hyperparameters['n_layers'])))
        if 'res_layers' in self.hyperparameters and self.hyperparameters['res_layers'] > 0:
            layer_summary.append('{:d}R'.format(int(self.hyperparameters['res_layers'])))
        if self.uses_bottleneck:
            layer_summary.append('F')
        if 'dropout' in self.hyperparameters:
            layer_summary.append('+D')
        if 'use_batchnorm' in self.hyperparameters and self.hyperparameters['use_batchnorm']:
            layer_summary.append('+BN')

        parameter_summary.append(''.join(layer_summary))
        parameter_summary.append('r:{}'.format(self.hyperparameters['rounding']))
        parameter_summary.append('Q+{}bpf'.format(self.latent_bpf) if self.train_codebook else 'Q-{}bpf'.format(self.latent_bpf))
        parameter_summary.append('S+' if self.scale_latent else 'S-')
        if self.entropy_weight is not None:
            parameter_summary.append('H+{:.2f}'.format(self.entropy_weight))

        return '{}/{}'.format(super().model_code, '-'.join(parameter_summary))
