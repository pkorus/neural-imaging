import numpy as np
import tensorflow as tf

from models.tfmodel import TFModel
from helpers import tf_helpers, paramspec


class DCN(TFModel):
    """
    An abstract class for deriving image compression models.

    # Attributes set-up by the abstract class:
      x                    - model input
      patch_size           - patch size
      latent_bpf           - number of bits per feature of the latent representation
      train_codebook       - whether the codebook
      codebook             - the quantization code book (TF)
      entropy_weight       - entropy regularization strength for model loss
      default_val_is_train - used to set default value for the 'is_training' flag during model inference
                             (useful for models with batch normalization)
      scale_latent         - bool flag indicating scaling of the latent representation
      use_batchnorm        - bool flag indicating the use of batch norm in the model

      weights              - soft quantization weights (TF)
      histogram            - latent space histogram based on soft quantization (TF)
      entropy              - entropy estimation (TF)

    # Attributes that need to be set-up by the derived classes:
      y
      latent_pre           - latent representation before quantization
      latent_post          - latent representation after quantization
      latent_shape         - shape of the latent tensor (before flattening)
      n_latent             - dimensionality of the latent representation
      _h                   - hyper parameters

    For setting up quantization, use the provided self._setup_latent_space method - it will create the latent_pre and
    latent_post attributes.
    """

    def __init__(self, sess, graph, label=None, x=None, nip_input=None, patch_size=128, latent_bpf=4, train_codebook=False, entropy_weight=None, default_val_is_train=True, scale_latent=False, use_batchnorm=False, use_gdn=False, verbose=False, loss_metric='L2', **kwargs):
        """
        Creates a forensic analysis network.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param label: a suffix for the name scope of the model
        """
        super().__init__(sess, graph, label)

        # Basic parameter sanitization

        if latent_bpf < 1 or latent_bpf > 8:
            raise ValueError('Invalid value for latent_bpf! Valid range: 1 - 8')

        if entropy_weight is not None and entropy_weight < 0:
            raise ValueError('Invalid value for entropy_weight! Valid range: >=0')

        self.verbose = verbose
        self.patch_size = patch_size
        self.nip_input = nip_input
        self.latent_bpf = latent_bpf
        self.train_codebook = train_codebook
        self.entropy_weight = entropy_weight
        self.default_val_is_train = default_val_is_train
        self.scale_latent = scale_latent
        self.use_batchnorm = use_batchnorm
        self.use_gdn = use_gdn
        self.loss_metric = loss_metric

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
            
            # Setup quantization code book -----------------------------------------------------------------------------
            with tf.name_scope('{}/optimization'.format(self.scoped_name)):
                
                with tf.name_scope('entropy'):
                                        
                    # Initialize the quantization codebook
                    qmin = -2 ** (self.latent_bpf - 1) + 1
                    qmax = 2 ** (self.latent_bpf - 1)
                                        
                    self.log('Initializing {} codebook ({} bpf): from {} to {}'.format('trainable' if self.train_codebook else 'fixed', self.latent_bpf, qmin, qmax))

                    if self.train_codebook:
                        bin_centers = tf.get_variable('{}/quantization/codebook'.format(self.scoped_name),
                            initializer=tf.constant_initializer(np.arange(qmin, qmax + 1)),
                            shape=(1, 2 ** self.latent_bpf))
                    else:
                        bin_centers = tf.constant(np.arange(qmin, qmax + 1), shape=(1, 2 ** self.latent_bpf), dtype=tf.float32)                        

                    self._codebook = bin_centers
            
            # Construct the actual model -------------------------------------------------------------------------------
            self.construct_model(kwargs)
            
            # Overwrite the output to guarantee correct data range and maintain gradient propagation
            self.y = tf.stop_gradient(tf.clip_by_value(self.y, 0, 1) - self.y) + self.y

            # Check if the sub-class has set all expected attributes
            setup_status = {key: hasattr(self, key) for key in ['y', 'latent_pre', 'latent_post', 'latent_shape', 'n_latent']}

            if not all(setup_status.values()):
                raise NotImplementedError('The model construction function has failed to set-up some attributes: {}'.format([key for key, value in setup_status.items() if not value]))

            # Add entropy estimation and model optimization operations -------------------------------------------------
            with tf.name_scope('{}/optimization'.format(self.scoped_name)):

                # Estimate entropy of the latent representation
                with tf.name_scope('entropy'):
                    self.entropy, self.histogram, self.weights = tf_helpers.entropy(self.latent_pre, self._codebook)

                # Loss and SSIM
                self.ssim = tf.reduce_mean(tf.image.ssim(self.x, tf.clip_by_value(self.y, 0, 1), max_val=1))
                
                if loss_metric == 'L2':                    
                    self.loss = tf.nn.l2_loss(self.x - self.y)
                else:
                    raise NotImplementedError('Loss metric {} not supported.'.format(loss_metric))
                
                if self.entropy_weight is not None:
                    self.loss = self.loss + self.entropy_weight * tf.cast(self.entropy, dtype=tf.float32)
                loss_entropy_label = '+ {:.2f} * entropy'.format(self.entropy_weight) if self.entropy_weight is not None else ''
                self.log('Initializing loss: {} {}'.format(self.loss_metric, loss_entropy_label))
                
                # Optimization
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.lr = tf.placeholder(tf.float32, name='{}_learning_rate'.format(self.scoped_name))
                    self.adam = tf.train.AdamOptimizer(learning_rate=self.lr)
                    self.opt = self.adam.minimize(self.loss, var_list=self.parameters)

    def log(self, message):
        if self.verbose:
            print(' ', message)

    def construct_model(self, params):
        raise NotImplementedError('Not implemented!')

    def _setup_latent_space(self, net):
        """
        Set up quantization of the latent space. The following attributes will be used (see constructor for details):
        - self.use_gdn
        - self.use_batchnorm
        - self.scale_latent
        - self._codebook
        - self._h.rounding

        The following new attributes will be set:
        - self.latent_pre (original real-values)
        - self.latent_post (quantized)

        :param net: the real-valued latent tensor
        :return: the quantized latent tensor
        """
        latent = tf.identity(net, name='{}/encoder/latent_raw'.format(self.scoped_name))

        # If requested, use GDN to Gaussianize the data
        if self.use_gdn:
            latent = tf.contrib.layers.GDN(latent)
            self.log('GDN: {}'.format(latent.shape))

        # If requested, add batch norm to normalize the latent representation
        if self.use_batchnorm:
            self.is_training = tf.placeholder(tf.bool, shape=(), name='{}/is_training'.format(self.scoped_name))
            latent = tf.contrib.layers.batch_norm(latent, scale=False, is_training=self.is_training,
                                                  scope='{}/encoder/bn_{}'.format(self.scoped_name, 0))
            self.log('batch norm: {}'.format(latent.shape))

        # Learn a scaling factor for the latent features to encourage greater values (facilitates quantization)
        if self.scale_latent:
            scaling_factor = 1
            alphas = tf.get_variable('{}/encoder/latent_scaling'.format(self.scoped_name), shape=(), dtype=tf.float32, initializer=tf.constant_initializer(scaling_factor))
            latent = tf.multiply(alphas, latent, name='{}/encoder/latent_scaled'.format(self.scoped_name))
            self.log('scaling latent representation - init:{}'.format(scaling_factor))

        # Add identity to facilitate better display in the TF graph
        latent = tf.identity(latent, name='{}/latent'.format(self.scoped_name))
        self.n_latent = int(np.prod(latent.shape[1:]))

        # Quantize the latent representation and remember tensors before and after the process
        self.latent_pre = latent
        latent = tf_helpers.quantization(latent, '{}/quantization'.format(self.scoped_name), 'latent_quantized',
                                         self._h.rounding, codebook_tensor=self._codebook)
        self.log('quantization with {} rounding'.format(self._h.rounding))
        self.latent_post = latent
        self.log('latent size: {} + quant:{}'.format(latent.shape, self._h.rounding))

        return latent

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'entropy': {'training': [], 'validation': []},
            'ssim': {'training': [], 'validation': []},
            'psnr': {'training': [], 'validation': []}
        }

    def get_tf_histogram(self, batch_x, is_training=None):
        with self.graph.as_default():
            feed_dict = {
                self.x if not self.use_nip_input else self.nip_input: batch_x,
            }

            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training if is_training is not None else self.default_val_is_train

            return self.sess.run(self.histogram, feed_dict=feed_dict)

    def compress(self, batch_x, is_training=None, direct=False):
        """
        Compress an input batch to a quantized latent representation.

        :param batch_x: Input tensor (N, H, W, 3:rgb) or (N, H, W, 4:rggb) for RAW data chained through a NIP
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :param direct: controls whether the input is a RAW image (chained through a NIP) or direct RGB input
        :return:
        """
        with self.graph.as_default():
            
            feed_dict = {
                self.x if (direct or not self.use_nip_input) else self.nip_input: batch_x,
            }            
            
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training if is_training is not None else self.default_val_is_train

            y = self.sess.run(self.latent_post, feed_dict=feed_dict)
            return y

    def compress_soft(self, batch_x, is_training=None, direct=False):
        """
        Compress an input batch to a pre-quantization real-valued latent representation.

        :param batch_x: Input tensor (N, H, W, 3:rgb) or (N, H, W, 4:rggb) for RAW data chained through a NIP
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :param direct: controls whether the input is a RAW image (chained through a NIP) or direct RGB input
        :return:
        """

        with self.graph.as_default():
            
            feed_dict = {
                self.x if (direct or not self.use_nip_input) else self.nip_input: batch_x,
            }
            
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training if is_training is not None else self.default_val_is_train
            
            y = self.sess.run(self.latent_pre, feed_dict=feed_dict)
            return y        
        
    def decompress(self, batch_z, is_training=None):
        """
        Decompress a batch of images from their quantized latent representations.
        :param batch_z: batch of quantized latent values
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :return:
        """
        with self.graph.as_default():
            
            feed_dict = {
                self.latent_post: batch_z
            }
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = 1.0

            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training if is_training is not None else self.default_val_is_train

            y = self.sess.run(self.y, feed_dict)
            return y.clip(0, 1)
            
    def process(self, batch_x, dropout_keep_prob=1.0, is_training=None, direct=False):
        """
        Process the image through the whole model (encoder-quantization-decoder).
        :param batch_x: Input tensor (N, H, W, 3:rgb) or (N, H, W, 4:rggb) for RAW data chained through a NIP
        :param dropout_keep_prob: set keep probability in case of using Dropout
        :param is_training: can be used to override the default 'is_training' flag (may be useful for models with BN)
        :param direct: controls whether the input is a RAW image (chained through a NIP) or direct RGB input
        """
        with self.graph.as_default():
            
            feed_dict={
                self.x if (direct or not self.use_nip_input) else self.nip_input: batch_x
            }
            
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = dropout_keep_prob
                
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training if is_training is not None else self.default_val_is_train
              
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
                'loss': np.sqrt(2 * loss),  # The L2 loss in TF is computed differently (half of non-square rooted norm)
                'ssim': ssim,
                'entropy': entropy
            }

    def compression_stats(self, patch_size=None, n_latent_bytes=None):
        """
        Get expected compression stats for the model:
            - data rate
            - bits per pixel (bpp)
            - bits per feature (bpf)
            - bytes

        :param patch_size: Can be used to override the default input size
        :param n_latent_bytes: Can be used to override the default bpf; Specified per feature.
        :return:
        """

        n_latent_bytes = n_latent_bytes or self.latent_bpf / 8

        ps = patch_size or self.patch_size        
        if ps is None:
            raise ValueError('Patch size not specified!')
            
        bitmap_size = ps * ps * 3
        return {
            'rate': bitmap_size / (n_latent_bytes * self.n_latent),
            'bpp': 8 * self.n_latent * n_latent_bytes / (ps * ps),
            'bpf': 8 * n_latent_bytes,
            'bytes': self.n_latent * n_latent_bytes
        }
    
    def summary(self):
        return 'dcn with {}-D latent representation [{:,} parameters]'.format(self.n_latent, self.count_parameters())
    
    @property
    def model_code(self):
        if not hasattr(self, 'n_latent'):
            raise ValueError('The model does not report the latent space dimensionality.')
        
        return '{}-{}D'.format(type(self).__name__, self.n_latent)        

    def get_parameters(self):
        return {
            'latent_bpf': self.latent_bpf,
            'train_codebook': self.train_codebook,
            'entropy_weight': self.entropy_weight,
            'default_val_is_train': self.default_val_is_train,
            'scale_latent': self.scale_latent,
            'use_batchnorm': self.use_batchnorm,
            'use_gdn': self.use_gdn
        }

    def get_codebook(self, bpf=None, lloyd=False):
        if hasattr(self, '_h') and hasattr(self._h, 'rounding'):

            bpf = bpf or self.latent_bpf

            if self._h.rounding in {'soft', 'identity'}:
                qmin = -2 ** (bpf - 1) + 1
                qmax = 2 ** (bpf - 1)
                return np.arange(qmin, qmax + 1).reshape((-1,))
            else:
                return self.sess.run(self._codebook).reshape((-1,))
        else:
            return self.sess.run(self._codebook).reshape((-1,))


class TwitterDCN(DCN):
    """
    Auto-encoder architecture described in:
    [1] L. Theis, W. Shi, A. Cunningham, and F. Huszár, “Lossy Image Compression with Compressive
    coders,” Mar. 2017.
    """

    def construct_model(self, params):

        # Define expected hyper parameters and their values ------------------------------------------------------------
        self._h = paramspec.ParamSpec({
            'n_features': (96, int, (4, 128)),
            'rounding': ('soft', str, {'identity', 'soft', 'soft-codebook', 'sin'}),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })

        self._h.update(**params)

        self.latent_shape = (1, self.patch_size // 8, self.patch_size // 8, self._h.n_features)
        self.n_latent = int(np.prod(self.latent_shape))

        activation = tf_helpers.activation_mapping[self._h.activation]

        self.log('Building Twitter DCN with d-latent={}'.format(self.n_latent))

        # Set-up op naming templates -----------------------------------------------------------------------------------
        le_norm = '{}/encoder/normalization'.format(self.scoped_name)
        le_conv = '{}/encoder/conv_{{}}'.format(self.scoped_name)
        le_relu = '{}/encoder/conv_{{}}/lrelu'.format(self.scoped_name)
        le_sum = '{}/encoder/sum_{{}}'.format(self.scoped_name)
        ld_norm = '{}/decoder/normalization'.format(self.scoped_name)
        ld_conv = '{}/decoder/conv_{{}}'.format(self.scoped_name)
        ld_tconv = '{}/decoder/tconv_{{}}'.format(self.scoped_name)
        ld_d2s = '{}/decoder/d2s_{{}}'.format(self.scoped_name)
        ld_sum = '{}/decoder/sum_{{}}'.format(self.scoped_name)

        # Encoder ------------------------------------------------------------------------------------------------------

        with tf.name_scope(le_norm):
            net = 2 * (self.x - 0.5)
            self.log('norm: {}'.format(net.shape))

        net = tf.contrib.layers.conv2d(net, 64, 5, 2, activation_fn=activation, scope=le_conv.format(0))
        self.log('conv:2 {} + {}'.format(net.shape, self._h.activation))
        net = tf.contrib.layers.conv2d(net, 128, 5, 2, activation_fn=None, scope=le_conv.format(1))
        self.log('conv:2 {} + {}'.format(net.shape, self._h.activation))

        net_relu = tf.nn.leaky_relu(net, name=le_relu.format(1))
        resnet = tf.contrib.layers.conv2d(net_relu, 128, 3, 1, activation_fn=activation, scope=le_conv.format(2))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, 1, activation_fn=None, scope=le_conv.format(3))
        net = tf.add(net, resnet, name=le_sum.format(0))
        self.log('res block {}'.format(net.shape))

        resnet = tf.contrib.layers.conv2d(net, 128, 3, 1, activation_fn=activation, scope=le_conv.format(4))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, 1, activation_fn=None, scope=le_conv.format(5))
        net = tf.add(net, resnet, name=le_sum.format(1))
        self.log('res block {}'.format(net.shape))

        resnet = tf.contrib.layers.conv2d(net, 128, 3, 1, activation_fn=activation, scope=le_conv.format(6))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, 1, activation_fn=None, scope=le_conv.format(7))
        net = tf.add(net, resnet, name=le_sum.format(2))
        self.log('res block {}'.format(net.shape))

        net = tf.contrib.layers.conv2d(net, self._h.n_features, 5, 2, activation_fn=None, scope=le_conv.format(8))
        self.log('conv:2 {} + {} activation'.format(net.shape, None))

        # Latent space -------------------------------------------------------------------------------------------------

        latent = self._setup_latent_space(net)

        # Decoder ------------------------------------------------------------------------------------------------------

        inet = tf.contrib.layers.conv2d(latent, 512, 3, 1, activation_fn=None, scope=ld_conv.format(0))
        self.log('conv:1 {} + {} activation'.format(inet.shape, None))
        inet = tf.depth_to_space(inet, 2, name=ld_d2s.format(0))
        self.log('dts {}'.format(inet.shape))

        resnet = tf.contrib.layers.conv2d(inet, 128, 3, 1, activation_fn=activation, scope=ld_conv.format(1))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, 1, activation_fn=None, scope=ld_conv.format(2))
        inet = tf.add(inet, resnet, name=ld_sum.format(0))
        self.log('res block {}'.format(inet.shape))

        resnet = tf.contrib.layers.conv2d(inet, 128, 3, 1, activation_fn=activation, scope=ld_conv.format(3))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, 1, activation_fn=None, scope=ld_conv.format(4))
        inet = tf.add(inet, resnet, name=ld_sum.format(1))
        self.log('res block {}'.format(inet.shape))

        resnet = tf.contrib.layers.conv2d(inet, 128, 3, 1, activation_fn=activation, scope=ld_conv.format(5))
        resnet = tf.contrib.layers.conv2d(resnet, 128, 3, 1, activation_fn=None, scope=ld_conv.format(6))
        inet = tf.add(inet, resnet, name=ld_sum.format(2))
        self.log('res block {}'.format(inet.shape))

        inet = tf.contrib.layers.conv2d(inet, 256, 3, 1, activation_fn=activation, scope=ld_tconv.format(7))
        self.log('conv:1 {} + {} activation'.format(inet.shape, self._h.activation))
        inet = tf.depth_to_space(inet, 2, name=ld_d2s.format(7))
        self.log('dts {}'.format(inet.shape))

        inet = tf.contrib.layers.conv2d(inet, 12, 3, 1, activation_fn=None, scope=ld_tconv.format(8))
        self.log('conv:1 {} + {} activation'.format(inet.shape, None))
        inet = tf.depth_to_space(inet, 2, name=ld_d2s.format(8))
        self.log('dts {}'.format(inet.shape))

        with tf.name_scope(ld_norm):
            y = (inet + 1) / 2
            self.log('denorm: {}'.format(y.shape))

        y = tf.identity(y, name="y")

        self.y = y
        self.latent = latent

    @property
    def model_code(self):
        parameter_summary = []

        if hasattr(self, 'latent_shape'):
            parameter_summary.append('x'.join(str(x) for x in self.latent_shape[1:]))

        parameter_summary.append('r:{}'.format(self._h.rounding))
        parameter_summary.append(
            'Q+{}bpf'.format(self.latent_bpf) if self.train_codebook else 'Q-{}bpf'.format(self.latent_bpf))
        parameter_summary.append('S+' if self.scale_latent else 'S-')
        if self.entropy_weight is not None:
            parameter_summary.append('H+{:.2f}'.format(self.entropy_weight))

        return '{}/{}'.format(super().model_code, '-'.join(parameter_summary))
