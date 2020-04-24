# -*- coding: utf-8 -*-
"""
Implementation of classic and neural image signal processors. The module provides:

- an abstract NIPModel class that sets up a common framework for ISP models
- Several neural ISPs: INet, DNet, UNet
- A trivial ONet model which serves as a NULL-ISP (allows to pass RGB-RGB pairs through the pipeline),
- A ClassicISP class with a standard ISP (standard steps + neural demosaicing)

"""
import os
import sys
import json
import inspect
import numpy as np
import tensorflow as tf

from collections import OrderedDict

import helpers.raw
from models.tfmodel import TFModel
from models import layers
from helpers import tf_helpers, paramspec, utils
from helpers.kernels import upsampling_kernel, gamma_kernels, bilin_kernel


class NIPModel(TFModel):
    """
    Abstract class for implementing neural imaging pipelines. Specific classes are expected to
    implement the 'construct_model' method that builds the model. See existing classes for examples.
    """

    def __init__(self, loss_metric='L2', patch_size=None, in_channels=4, **kwargs):
        """
        Base constructor with common setup.

        :param loss_metric: loss metric for NIP optimization (L2, L1, SSIM)
        :param patch_size: Optionally patch size can be given to fix placeholder dimensions (can be None)
        :param in_channels: number of channels in the input RAW image (defaults to 4 for RGGB)
        :param kwargs: Additional arguments for specific NIP implementations
        """
        super().__init__()
        self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, in_channels), name='x')
        self.in_channels = in_channels
        self.construct_model(**kwargs)
        self._has_attributes(['y', '_model'])

        # Configure loss and model optimization
        self.loss_metric = loss_metric
        self.construct_loss(loss_metric)
        self.optimizer = tf.keras.optimizers.Adam()

    def construct_loss(self, loss_metric):
        if loss_metric == 'L2':
            self.loss = tf_helpers.mse
        elif loss_metric == 'L1':
            self.loss = tf_helpers.mae
        elif loss_metric == 'SSIM':
            self.loss = tf_helpers.ssim_loss
        elif loss_metric == 'MS-SSIM':
            self.loss = tf_helpers.msssim_loss
        else:
            raise ValueError('Unsupported loss metric!')

    def construct_model(self):
        """
        Constructs the NIP model. The model should be a tf.keras.Model instance available via the
        self._model attribute. The method should use self.x as RAW image input, and set self.y as
        the model output. The output is expected to be clipped to [0,1]. For better optimization
        stability, it's better not to backpropagate through clipping:

        self.y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y])
        """
        raise NotImplementedError()

    def training_step(self, batch_x, batch_y, learning_rate=None):
        """
        Make a single training step and return the loss.
        """
        with tf.GradientTape() as tape:

            batch_Y = self._model(batch_x)
            loss = self.loss(batch_Y, batch_y)

        if learning_rate is not None: self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)

        if any(np.sum(np.isnan(x)) > 0 for x in grads):
            raise RuntimeError('∇ NaNs: {}'.format({p.name: np.mean(np.isnan(x)) for x, p in zip(grads, self._model.trainable_weights)}))

        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return loss.numpy()

    def process(self, batch_x, training=False):
        """
        Develop RAW input and return RGB image.
        """
        if batch_x.ndim == 3:
            batch_x = np.expand_dims(batch_x, 0)

        return self._model(batch_x, training)

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'psnr': {'validation': []},
            'ssim': {'validation': []},
            'dmse': {'validation': []}
        }

    def get_hyperparameters(self):
        p = {'in_channels': self.in_channels}
        if hasattr(self, '_h'):
            p.update(self._h.to_json())
        return p

    @property
    def _input_description(self):
        return utils.format_patch_shape(self.patch_size_raw)

    @property
    def _output_description(self):
        return utils.format_patch_shape(self.patch_size_rgb)

    @property
    def patch_size_raw(self):
        return self.x.shape[1:] if hasattr(self.y, 'shape') else None

    @property
    def patch_size_rgb(self):
        return self.y.shape[1:] if hasattr(self.y, 'shape') else None

    def summary(self):
        return '{:s} : {} -> {}'.format(super().summary(), self._input_description, self._output_description)

    def load_model(self, dirname):
        if '/' not in dirname:
            dirname = os.path.join('data/models/nip', dirname)
        super().load_model(dirname)

    def save_model(self, dirname, epoch=0, quiet=False):
        if '/' not in dirname:
            dirname = os.path.join('data/models/nip', dirname)
        super().save_model(dirname, epoch=epoch, quiet=quiet)


class UNet(NIPModel):
    """
    The UNet model, rewritten from scratch for TF 2.x
    Originally adapted from https://github.com/cchen156/Learning-to-See-in-the-Dark
    """

    def construct_model(self, **kwargs):
        # Define and validate hyper-parameters
        self._h = paramspec.ParamSpec({
            'n_steps': (5, int, (2, 6)),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })

        self._h.update(**kwargs)
        lrelu = tf_helpers.activation_mapping[self._h.activation]

        _layers = OrderedDict()
        _tensors = OrderedDict()
        _tensors['ep0'] = self.x

        # Construct the encoder
        for n in range(1, self._h.n_steps + 1):
            _layers['ec{}1'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(n-1), [3, 3], activation=lrelu, padding='SAME')
            _layers['ec{}2'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(n-1), [3, 3], activation=lrelu, padding='SAME')
            _tensors['ec{}1'.format(n)] = _layers['ec{}1'.format(n)](_tensors['ep{}'.format(n-1)])
            _tensors['ec{}2'.format(n)] = _layers['ec{}2'.format(n)](_tensors['ec{}1'.format(n)])

            if n < self._h.n_steps:
                _layers['ep{}'.format(n)] = tf.keras.layers.MaxPool2D([2, 2], padding='SAME')
                _tensors['ep{}'.format(n)]  = _layers['ep{}'.format(n)](_tensors['ec{}2'.format(n)])

        # Easy access to encoder output via a recursive relation
        _tensors['dc02'] = _tensors['ec{}2'.format(self._h.n_steps)]

        # Construct the decoder
        for n in range(1, self._h.n_steps):
            _layers['dct{}'.format(n)] = tf.keras.layers.Conv2DTranspose(32 * 2**(self._h.n_steps - n - 1), [2, 2], [2, 2], padding='SAME')
            _layers['dcat{}'.format(n)] = tf.keras.layers.Concatenate()
            _layers['dc{}1'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(self._h.n_steps - n - 1), [3, 3], activation=lrelu, padding='SAME')
            _layers['dc{}2'.format(n)] = tf.keras.layers.Conv2D(32 * 2**(self._h.n_steps - n - 1), [3, 3], activation=lrelu, padding='SAME')

            _tensors['dct{}'.format(n)] = _layers['dct{}'.format(n)](_tensors['dc{}2'.format(n-1)])
            _tensors['dcat{}'.format(n)] = _layers['dcat{}'.format(n)]([_tensors['dct{}'.format(n)], _tensors['ec{}2'.format(self._h.n_steps - n)]])
            _tensors['dc{}1'.format(n)] = _layers['dc{}1'.format(n)](_tensors['dcat{}'.format(n)])
            _tensors['dc{}2'.format(n)] = _layers['dc{}2'.format(n)](_tensors['dc{}1'.format(n)])

        # Final step to render the RGB image
        _layers['dc{}'.format(self._h.n_steps)] = tf.keras.layers.Conv2D(12, [3, 3], padding='SAME')
        _tensors['dc{}'.format(self._h.n_steps)] =_layers['dc{}'.format(self._h.n_steps)](_tensors['dc{}2'.format(self._h.n_steps - 1)])
        _tensors['dts'] = tf.nn.depth_to_space(_tensors['dc{}'.format(self._h.n_steps)], 2)

        # Add NIP outputs
        y = _tensors['dts']
        # self.y = tf.clip_by_value(_tensors['dts'], 0, 1)
        self.y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y

        # Construct the Keras model
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y], name='unet')

    @property
    def model_code(self):
        return f'{self.class_name}_{self._h.n_steps}'


class INet(NIPModel):
    """
    A neural pipeline which replicates the steps of a standard imaging pipeline.
    """

    def construct_model(self, random_init=False, kernel=5, trainable_upsampling=False, cfa_pattern='gbrg'):

        self._h = paramspec.ParamSpec({
            'random_init': (False, bool, None),
            'kernel': (5, int, (3, 11)),
            'trainable_upsampling': (False, bool, None),
            'cfa_pattern': ('gbrg', str, {'gbrg', 'rggb', 'bggr'})
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys() if k in params})

        # Initialize the upsampling kernel
        upk = upsampling_kernel(self._h.cfa_pattern)

        if self._h.random_init:
            # upk = np.random.normal(0, 0.1, (4, 12))
            dmf = np.random.normal(0, 0.1, (self._h.kernel, self._h.kernel, 3, 3))
            gamma_d1k = np.random.normal(0, 0.1, (3, 12))
            gamma_d1b = np.zeros((12, ))
            gamma_d2k = np.random.normal(0, 0.1, (12, 3))
            gamma_d2b = np.zeros((3,))
            srgbk = np.eye(3)
        else:
            # Prepare demosaicing kernels (bilinear)
            dmf = bilin_kernel(self._h.kernel)

            # Prepare gamma correction kernels (obtained from a pre-trained toy model)
            gamma_d1k, gamma_d1b, gamma_d2k, gamma_d2b = gamma_kernels()

            # Example sRGB conversion table
            srgbk = np.array([[ 1.82691061, -0.65497452, -0.17193617],
                                [-0.00683982,  1.33216381, -0.32532394],
                                [ 0.06269717, -0.40055895,  1.33786178]]).transpose()

        # Up-sample the input back the full resolution
        h12 = tf.keras.layers.Conv2D(12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False, activation=None, trainable=self._h.trainable_upsampling)(self.x)

        # Demosaicing
        pad = (self._h.kernel - 1) // 2
        bayer = tf.nn.depth_to_space(h12, 2)
        bayer = tf.pad(bayer, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
        rgb = tf.keras.layers.Conv2D(3, self._h.kernel, kernel_initializer=tf.constant_initializer(dmf), use_bias=False, activation=None, padding='VALID')(bayer)

        # Color space conversion
        srgb = tf.keras.layers.Conv2D(3, 1, kernel_initializer=tf.constant_initializer(srgbk), use_bias=False, activation=None)(rgb,)

        # Gamma correction
        rgb_g0 = tf.keras.layers.Conv2D(12, 1, kernel_initializer=tf.constant_initializer(gamma_d1k), bias_initializer=tf.constant_initializer(gamma_d1b), use_bias=True, activation=tf.keras.activations.tanh)(srgb)
        y = tf.keras.layers.Conv2D(3, 1, kernel_initializer=tf.constant_initializer(gamma_d2k), bias_initializer=tf.constant_initializer(gamma_d2b), use_bias=True, activation=None)(rgb_g0)

        # self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))
        self.y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y])

    @property
    def model_code(self):
        return '{c}_{cfa}{tu}{r}_{k}x{k}'.format(c=self.class_name, cfa=self._h.cfa_pattern, k=self._h.kernel,
            tu='T' if self._h.trainable_upsampling else '', r='R' if self._h.random_init else '')


class DNet(NIPModel):
    """
    Neural imaging pipeline adapted from a joint demosaicing-&-denoising model:
    Gharbi, Michaël, et al. "Deep joint demosaicking and denoising." ACM Transactions on Graphics (TOG) 35.6 (2016): 191.
    """

    def construct_model(self, n_layers=15, kernel=3, n_features=64):

        self._h = paramspec.ParamSpec({
            'n_layers': (15, int, (1, 32)),
            'kernel': (3, int, (3, 11)),
            'n_features': (64, int, (4, 128)),
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys() if k in params})

        k_initializer = tf.keras.initializers.VarianceScaling

        # Initialize the upsampling kernel
        upk = upsampling_kernel()

        # Padding size
        pad = (self._h.kernel - 1) // 2

        # Convolutions on the sub-sampled input tensor
        deep_x = self.x
        for r in range(self._h.n_layers):
            deep_y = tf.keras.layers.Conv2D(12 if r == self._h.n_layers - 1 else self._h.n_features, self._h.kernel, activation=tf.keras.activations.relu, padding='VALID', kernel_initializer=k_initializer)(deep_x)
            deep_x = tf.pad(deep_y, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')

        # Up-sample the input
        h12 = tf.keras.layers.Conv2D(12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False, activation=None, trainable=False)(self.x)
        bayer = tf.nn.depth_to_space(h12, 2)

        # Upscale the conv. features and concatenate with the input RGB channels
        features = tf.nn.depth_to_space(deep_x, 2)
        bayer_features = tf.concat((features, bayer), axis=3)

        # Project the concatenated 6-D features (R G B bayer from input + 3 channels from convolutions)
        pu = tf.keras.layers.Conv2D(self._h.n_features, self._h.kernel, kernel_initializer=k_initializer, use_bias=True, activation=tf.keras.activations.relu, padding='VALID', bias_initializer=tf.zeros_initializer)(bayer_features)

        # Final 1x1 conv to project each 64-D feature vector into the RGB colorspace
        pu = tf.pad(pu, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')

        y = tf.keras.layers.Conv2D(3, 1, kernel_initializer=tf.ones_initializer, use_bias=False, activation=None, padding='VALID')(pu)
        # self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.scoped_name))
        self.y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        self._model = tf.keras.Model(inputs=[self.x], outputs=[self.y])

    @property
    def model_code(self):
        return '{c}_{k}x{k}_{l}x{f}f'.format(c=self.class_name, k=self._h.kernel,
            f=self._h.n_features, l=self._h.n_layers)


class ONet(NIPModel):
    """
    Dummy pipeline for RGB training.
    """

    def construct_model(self):
        patch_size = 2 * self.x.shape[1]
        self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, 3))
        self.y = tf.identity(self.x)
        self._model = tf.keras.Model(inputs=self.x, outputs=self.y)


class __TensorISP():
    """
    Toy ISP implemented in Tensorflow. This class is intended for debugging and testing - for
    use in most situations, please use a more flexible 'ClassicISP' which integrates with
    the rest of the framework.
    """

    def process(self, x, srgb_mat=None, cfa_pattern='gbrg', brightness='percentile'):

        kernel = 5

        # Initialize upsampling and demosaicing kernels
        upk = upsampling_kernel(cfa_pattern).reshape((1, 1, 4, 12))
        dmf = bilin_kernel(kernel)

        # Setup sRGB color conversion
        if srgb_mat is None:
            srgb_mat = np.eye(3)
        srgb_mat = srgb_mat.T.reshape((1, 1, 3, 3))

        # Demosaicing & color space conversion
        pad = (kernel - 1) // 2
        h12 = tf.nn.conv2d(x, upk, [1, 1, 1, 1], 'SAME')
        bayer = tf.nn.depth_to_space(h12, 2)
        bayer = tf.pad(bayer, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
        rgb = tf.nn.conv2d(bayer, dmf, [1, 1, 1, 1], 'VALID')

        # RGB -> sRGB
        rgb = tf.nn.conv2d(rgb, srgb_mat, [1, 1, 1, 1], 'SAME')

        # Brightness correction
        if brightness is not None:
            if brightness == 'percentile':
                percentile = 0.5
                rgb -= np.percentile(rgb, percentile)
                rgb /= np.percentile(rgb, 100 - percentile)
            elif brightness == 'shift':
                mult = 0.25 / tf.reduce_mean(rgb)
                rgb *= mult
            else:
                raise ValueError('Brightness normalization not recognized!')

        # Gamma correction
        y = rgb
        y = tf.stop_gradient(tf.clip_by_value(y, 0, 1) - y) + y
        y = tf.pow(y, 1/2.2)

        return y


class _ClassicISP(tf.keras.Model):
    """
    A flexible version of a classic camera ISP.
    """

    def __init__(self, srgb_mat=None, kernel=5, c_filters=(3,), cfa_pattern='gbrg', residual=False, brightness=None, **kwargs):
        super().__init__()

        up = upsampling_kernel(cfa_pattern).reshape((1, 1, 4, 12)).astype(np.float32)
        self._upsampling_kernel = tf.convert_to_tensor(up)

        if srgb_mat is None:
            srgb_mat = np.eye(3, dtype=np.float32)

        self._srgb_mat = tf.convert_to_tensor(srgb_mat.T.reshape((1, 1, 3, 3)))
        self._demosaicing = layers.DemosaicingLayer(c_filters, kernel, 'leaky_relu', residual)
        self._brightness = brightness

    def call(self, inputs, training=False):
        h12 = tf.nn.conv2d(inputs, self._upsampling_kernel, [1, 1, 1, 1], 'SAME')
        bayer = tf.nn.depth_to_space(h12, 2)

        rgb = self._demosaicing(bayer)
        rgb = tf.nn.conv2d(rgb, self._srgb_mat, [1, 1, 1, 1], 'SAME')

        # Brightness correction
        if self._brightness == 'percentile':
            percentile = 0.5
            rgb -= np.percentile(rgb, percentile)
            rgb /= np.percentile(rgb, 100 - percentile)
        elif self._brightness == 'shift':
            mult = 0.25 / tf.reduce_mean(rgb)
            rgb *= mult

        # Gamma correction
        y = rgb
        y = tf.stop_gradient(tf.clip_by_value(y, 1.0/255, 1) - y) + y
        y = tf.pow(y, 1/2.2)
        return y


class ClassicISP(NIPModel):
    """
    A tensorflow implementation of a simple camera ISP. The model expects RAW Bayer stacks
    with 4 channels (RGGB) as input, and replicates steps of a simple pipeline:

    - upsample half-resolution RGGB stacks to full-resolution RGB Bayer images
    - demosaicing (simple CNN model)
    - RGB -> sRGB color conversion (based on conversion tables from the camera)
    - [optional brightness normalization]
    - gamma correction

    See also: helpers.raw_api.unpack
    """

    def construct_model(self, srgb_mat=None, kernel=5, c_filters=(), cfa_pattern='gbrg', residual=True, brightness=None):

        self._h = paramspec.ParamSpec({
            'kernel': (5, int, (3, 11)),
            'c_filters': ((), tuple, paramspec.numbers_in_range(int, 1, 1024)),
            'cfa_pattern': ('gbrg', str, {'gbrg', 'rggb', 'bggr'}),
            'residual': (True, bool, None)
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys() if k in params})
        self._model = _ClassicISP(**self._h.to_dict())
        self.y = None

    def set_cfa_pattern(self, cfa_pattern):
        if cfa_pattern is not None:
            cfa_pattern = cfa_pattern.lower()
            up = upsampling_kernel(cfa_pattern).reshape((1, 1, 4, 12)).astype(np.float32)
            self._model._upsampling_kernel = tf.convert_to_tensor(up)
            self._h.update(cfa_pattern=cfa_pattern)

    def set_srgb_conversion(self, srgb_mat):
        if srgb_mat is not None:
            srgb = srgb_mat.T.reshape((1, 1, 3, 3)).astype(np.float32)
            self._model._srgb_mat = tf.convert_to_tensor(srgb)

    def process(self, batch_x, training=False, cfa_pattern=None, srgb_mat=None):
        if batch_x.ndim == 3:
            batch_x = np.expand_dims(batch_x, 0)

        self.set_cfa_pattern(cfa_pattern)
        self.set_srgb_conversion(srgb_mat)
        return self._model(batch_x, training)

    @property
    def model_code(self):
        return 'ClassicISP_{cfa}_{k}x{k}_{fs}-{of}{r}'.format(fs='-'.join(['{:d}'.format(x) for x in self._h.c_filters]), of=3, k=self._h.kernel, cfa=self._h.cfa_pattern, r='R' if self._h.residual else '')

    def set_camera(self, camera):
        """ Sets both CFA and sRGB based on camera presets from 'config/cameras.json' """
        with open('config/cameras.json') as f:
            cameras = json.load(f)
        self.set_cfa_pattern(cameras[camera]['cfa'])
        self.set_srgb_conversion(np.array(cameras[camera]['srgb']))

    @classmethod
    def restore(cls, dir_name='data/models/isp/ClassicISP_auto_3x3_32-32-32-32-3R/', *, camera=None, cfa=None, srgb=None, patch_size=128):
        isp = super().restore(dir_name)

        if camera is not None:
            isp.set_camera(camera)

        if cfa is not None:
            isp.set_cfa_pattern(cfa)

        if srgb is not None:
            isp.set_srgb_conversion(cfa)

        return isp

    def summary(self):
        nf = len(self._h.c_filters)
        fs = self._h.c_filters[0] if len(set(self._h.c_filters)) == 1 else '*'
        k = self._h.kernel
        return f'{self.class_name}[{self._h.cfa_pattern}] + CNN demosaicing [{nf}+1 layers : {k}x{k}x{fs} -> 1x1x3]'

    def summary_compact(self):
        nf = len(self._h.c_filters)
        fs = self._h.c_filters[0] if len(set(self._h.c_filters)) == 1 else '*'
        k = self._h.kernel
        return f'{self.class_name}[{self._h.cfa_pattern}, {nf}+1 conv2D {k}x{k}x{fs} > 1x1x3]'


supported_models = [name for name, obj in inspect.getmembers(sys.modules[__name__]) if type(obj) is type and issubclass(obj, NIPModel) and name != 'NIPModel']
