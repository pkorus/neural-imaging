# -*- coding: utf-8 -*-
"""
Models for learned image compression.

This module implements the machine learning models (useful for building other models/workflows). For use as an actual
codec (with full encoding to/from bitstreams) see the 'compression.codec' module. The 'codec' module also simplifies
restoring some of the provided baseline models, e.g.:

    from compression import codec
    codec.restore('16c')

Instead of the standard:

    a) tfmodel.restore('data/models/dcn/baselines/16c/', compression, key='codec')
    b) compression.TwitterDCN.restore('data/models/dcn/baselines/16c/', key='codec')

The provided abstract class (DCN) can be used for deriving new models. The default learned codec available with the
toolbox is TwitterDCN.
"""
import numpy as np
import tensorflow as tf

from models.layers import DiscreteLatent
from models.tfmodel import TFModel
from helpers import tf_helpers, paramspec


class DCN(TFModel):
    """
    An abstract class for learned image compression models. Override 'construct_model' in child classes to 
    create specific models.

    # Attributes set-up by the abstract class:
      x                    - model input
      _h                   - basic hyper-parameters - see helpers.ParamSpec
      ssim                 - function to compute the SSIM
      loss                 - function to compute the loss
      optimizer            - TF optimizer

    # Attributes that need to be set-up by the derived classes:
      y                    - output tensor
      _model               - tf.keras.Model of the entire codec (for training)
      _encoder             - tf.keras.Model of just the encoder
      _decoder             - tf.keras.Model of just the decoder

    For setting up quantization, use the provided DiscreteLatent layer (self.discrete_latent).
    """

    def __init__(self, patch_size=128, latent_bpf=5, rounding='soft-codebook', train_codebook=False,
                 entropy_weight=250, scale_latent=True, use_batchnorm=False, loss_metric='L2', **kwargs):
        """
        :param label: A suffix to the scoped name (used when saving the model)
        :param patch_size: patch size, specify the number to access model statistics (can be set to None)
        :param latent_bpf: precision of latent space quantization (in bits per feature)
        :param rounding: rounding method, best to use the default 'soft-codebook'
        :param train_codebook: set to true to make the quantization codebook trainable (not tested)
        :param entropy_weight: weight of the entropy term in the training loss
        :param scale_latent: whether to use a trainable scaling factor for the latent representation (typically needed)
        :param use_batchnorm: self-explanatory, currently not used, intended for future models
        :param loss_metric: currently not used, only L2 (with entropy regularization) is implemented 
        :param **kwargs: additional arguments for child classes
        """
        super().__init__()

        # Parameter sanitization
        self._h = paramspec.ParamSpec({
            'latent_bpf': (5, int, (1, 8)),
            'train_codebook': (False, bool, None),
            'entropy_weight': (250, float, (0, 1e6)),
            'scale_latent': (True, bool, None),
            'use_batchnorm': (False, bool, None),
            'loss_metric': ('L2', str, {'L2'}),
            'rounding': ('soft', str, {'identity', 'soft', 'soft-codebook', 'sin'})
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})
        self.patch_size = patch_size

        self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, 3))

        # Prepare the quantization layer        
        self.discrete_latent = DiscreteLatent(self._h.rounding, latent_bpf=self._h.latent_bpf)

        # Construct the neural network model
        self.construct_model(**kwargs)
        self._has_attributes(['y', '_model', '_encoder', '_decoder'])
        
        # Loss and SSIM
        self.ssim = lambda a, b: tf.reduce_mean(tf.image.ssim(a, b, max_val=1))
        
        if loss_metric == 'L2':
            def mse_entropy(image_target, image_compressed, entropy):
                return tf.nn.l2_loss(image_target - image_compressed) + self._h.entropy_weight * entropy
            self.loss = mse_entropy
        else:
            raise NotImplementedError('Loss metric {} not supported yet.'.format(loss_metric))
                    
        # Optimization
        self.optimizer = tf.keras.optimizers.Adam()

    def construct_model(self, params):
        raise NotImplementedError('Not implemented!')

    def reset_performance_stats(self):
        self.performance = self._reset_performance(['loss', 'entropy', 'ssim', 'psnr'])

    def compress(self, batch_x):
        """ Compress an input batch (NHW3:rgb) to a quantized latent representation. """
        return self._encoder(np.expand_dims(batch_x, axis=0) if batch_x.ndim == 3 else batch_x)[0]
        
    def decompress(self, batch_z):
        """ Decompress a batch of images from their quantized latent representations. """
        return self._decoder(np.expand_dims(batch_z, axis=0) if batch_z.ndim == 3 else batch_z)
            
    def process(self, batch_x, return_entropy=False):
        """ Process a batch of images (NHW3:rgb) through the entire model (encoder-quantization-decoder). """
        batch_y, entropy = self._model(batch_x)
        if return_entropy:
            return batch_y, entropy
        else:
            return batch_y

    def training_step(self, batch_x, learning_rate=None):
        """ Make a single training step and return the current loss. """
        with tf.GradientTape() as tape:
            batch_Y, entropy = self._model(batch_x)
            loss = self.loss(batch_x, batch_Y, entropy)
        
        ssim = self.ssim(tf.convert_to_tensor(batch_x), tf.convert_to_tensor(batch_Y))

        if learning_rate is not None: self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return {
                'loss': np.sqrt(2 * loss),  # The L2 loss in TF is computed differently (half of non-square rooted norm)
                'ssim': ssim,
                'entropy': entropy
            }

    def compression_stats(self, patch_size=None, n_latent_bytes=None):
        """
        Get expected compression stats:
            - data rate
            - bits per pixel (bpp)
            - bits per feature (bpf)
            - bytes

        :param patch_size: Can be used to override the default input size
        :param n_latent_bytes: Can be used to override the default bpf; Specified in bytes per feature.
        :return:
        """

        n_latent_bytes = n_latent_bytes or self._h.latent_bpf / 8

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
        l_shape = 'x'.join(str(x) for x in self.latent_shape if x is not None)
        bpf = self._h.latent_bpf
        params = self.count_parameters()
        return f'{self.class_name} : {l_shape}-D latent space @ {bpf}-bpf [{params:,.0f} params]'

    def summary_compact(self):
        return f'{self.class_name} {self.latent_shape[-1]}-D'

    @property
    def model_code(self):
        if not hasattr(self, 'n_latent'):
            raise ValueError('The model does not report the latent space dimensionality.')
        
        return '{}-{}C'.format(type(self).__name__, self._h.n_features)        

    def get_codebook(self):
        return self.discrete_latent.quantization.codebook.numpy().reshape((-1,))


class TwitterDCN(DCN):
    """
    Adaptation of the auto-encoder architecture described in:
    [1] Theis, Shi, Cunningham & Huszár, “Lossy Image Compression with Compressive coders,” Mar. 2017.

    # Extra-hyperparameters
      n_features: number of features in the latent representation (per spatial location)
      activation: activation function (see helpers.tf_helpers.activation_mapping for available activations)
    """

    def construct_model(self, n_features=32, activation='leaky_relu'):

        # Define expected hyper parameters and their values ------------------------------------------------------------
        self._h.add({
            'n_features': (32, int, (4, 128)),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })

        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys() if k in params})

        if self.patch_size is None:
            self.latent_shape = (None, None, self._h.n_features)
            self.n_latent = None
        else:
            self.latent_shape = (self.patch_size // 8, self.patch_size // 8, self._h.n_features)
            self.n_latent = int(np.prod(self.latent_shape))

        activation = tf_helpers.activation_mapping[self._h.activation]

        # Encoder ------------------------------------------------------------------------------------------------------

        net = 2 * (self.x - 0.5)

        net = tf.keras.layers.Conv2D(64, 5, 2, padding='SAME', activation=activation)(net)
        net = tf.keras.layers.Conv2D(128, 5, 2, padding='SAME', activation=None)(net)

        net_relu = tf.nn.leaky_relu(net)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(net_relu)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        net = tf.add(net, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(net)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        net = tf.add(net, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(net)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        net = tf.add(net, resnet)

        net = tf.keras.layers.Conv2D(self._h.n_features, 5, 2, padding='SAME', activation=None)(net)

        # Latent space -------------------------------------------------------------------------------------------------

        self.latent, self.entropy = self.discrete_latent(net)

        # Decoder ------------------------------------------------------------------------------------------------------

        self.latent_input = tf.keras.Input(dtype=tf.float32, shape=self.latent.shape[1:])

        inet = tf.keras.layers.Conv2D(512, 3, 1, padding='SAME', activation=None)(self.latent_input)
        inet = tf.nn.depth_to_space(inet, 2)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(inet)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        inet = tf.add(inet, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(inet)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        inet = tf.add(inet, resnet)

        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=activation)(inet)
        resnet = tf.keras.layers.Conv2D(128, 3, 1, padding='SAME', activation=None)(resnet)
        inet = tf.add(inet, resnet)

        inet = tf.keras.layers.Conv2D(256, 3, 1, padding='SAME', activation=activation)(inet)
        inet = tf.nn.depth_to_space(inet, 2)

        inet = tf.keras.layers.Conv2D(12, 3, 1, padding='SAME', activation=None)(inet)
        inet = tf.nn.depth_to_space(inet, 2)

        y = (inet + 1) / 2

        # Overwrite the output to guarantee correct data range and maintain gradient propagation
        self.y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        
        # Create separate models to enable separate encoding / decoding steps
        self._encoder = tf.keras.Model(inputs=self.x, outputs=[self.latent, self.entropy], name='encoder')
        self._decoder = tf.keras.Model(inputs=self.latent_input, outputs=self.y, name='decoder')

        # Combine the models to enable compression simulation, training and 1-step model saving / loading
        latent, entropy = self._encoder(self.x)
        self._model = tf.keras.Model(inputs=self.x, outputs=[self._decoder(latent), entropy], name='codec')

    @property
    def model_code(self):
        parameter_summary = []
        parameter_summary.append(self._h.rounding)
        parameter_summary.append(
            f'Q+{self._h.latent_bpf}bpf' if self._h.train_codebook else f'Q-{self._h.latent_bpf}bpf')
        parameter_summary.append('S+' if self._h.scale_latent else 'S-')
        if self._h.entropy_weight is not None:
            parameter_summary.append(f'H+{self._h.entropy_weight:.2f}')

        return f'{super().model_code}/{"_".join(parameter_summary)}'
