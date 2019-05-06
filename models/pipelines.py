import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from models.tfmodel import TFModel
from helpers.utils import upsampling_kernel, bilin_kernel, gamma_kernels, lrelu, upsample_and_concat


class NIPModel(TFModel):
    """
    Abstract class for implementing neural imaging pipelines. Specific classes are expected to implement the
    'construct_model' method that builds the model, and 'parameters' method which lists its parameters. See existing
    classes for examples.
    """

    def __init__(self, sess=None, graph=None, loss_metric='L2', patch_size=None, label=None, reuse_placeholders=None, **kwargs):        
        """
        Base constructor with common setup.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param loss_metric: loss metric for NIP optimization (L2, L1, SSIM)
        :param patch_size: Optionally patch size can be given to fix placeholder dimensions (can be None)
        :param label: A string prefix for the model (useful when multiple NIPs are used in a single TF graph)
        :param reuse_placeholders: Give a dictionary with 'x' and 'y' keys if multiple NIPs should use the same inputs
        :param kwargs: Additional arguments for specific NIP implementations
        """
        super().__init__(sess, graph, label)

        # Initialize input placeholders and run 'construct_model' to build the model and
        # setup its output as self.y
        self.y = None  # This will be set up later by child classes

        if reuse_placeholders is not None:
            self.x = reuse_placeholders['x']
            self.y_gt = reuse_placeholders['y']
        else:
            with self.graph.as_default():
                self.x = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 4), name='x')
                self.y_gt = tf.placeholder(tf.float32, shape=(None, 2 * patch_size if patch_size is not None else None, 2 * patch_size if patch_size is not None else None, 3), name='y')
        
        self.label = '_'+label if label is not None else ''
        self.construct_model(**kwargs)

        # Configure loss and model optimization
        self.loss_metric = loss_metric
        self.construct_loss(loss_metric)

    def construct_loss(self, loss_metric):
        with self.graph.as_default():
            with tf.name_scope('nip_optimization'):
                # Detect whether non-clipped image is available (better training stability)
                y = self.yy if hasattr(self, 'yy') else self.y
                
                # The loss
                if loss_metric == 'L2':
                    self.loss = tf.reduce_mean(tf.pow(255.0*y - 255.0*self.y_gt, 2.0))
                elif loss_metric == 'L1':
                    self.loss = tf.reduce_mean(tf.abs(255.0*y - 255.0*self.y_gt))
                elif loss_metric == 'SSIM':
                    self.loss = 255 * (1 - tf.image.ssim_multiscale(y, self.y_gt, 1.0))
                else:
                    raise ValueError('Unsupported loss metric!')

                # In case the model used batch norm
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):                
                    
                    # Learning rate
                    self.lr = tf.placeholder(tf.float32, name='nip_learning_rate')

                    # Create the optimizer and make sure only the parameters of the current model are updated
                    self.adam = tf.train.AdamOptimizer(learning_rate=self.lr, name='nip_adam{}'.format(self.label))
                    self.opt = self.adam.minimize(self.loss, var_list=self.parameters)
    
    def construct_model(self):
        """
        Constructs the NIP model. The method should use self.x as RAW image input, and set self.y as the model output.
        The output is expected to be clipped to [0,1]. For better optimization stability, the model can set self.yy to
        non-clipped output (will be used for gradient computation).

        A string prefix (self.label) should be used for variables / named scopes to facilitate using multiple NIPs
        in a single TF graph.
        """
        raise NotImplementedError()

    def training_step(self, batch_x, batch_y, learning_rate):
        """
        Make a single training step and return the loss.
        """
        with self.graph.as_default():
            feed_dict={
                    self.x: batch_x,
                    self.y_gt: batch_y,
                    self.lr: learning_rate
            }
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = True
                
            _, loss = self.sess.run([self.opt, self.loss], feed_dict=feed_dict)
            return loss
        
    def process(self, batch_x, is_training=False):
        """
        Develop RAW input and return RGB image.
        """
        if batch_x.ndim == 3:
            batch_x = np.expand_dims(batch_x, 0)
        
        with self.graph.as_default():
            feed_dict={self.x: batch_x}
            if hasattr(self, 'is_training'):
                feed_dict[self.is_training] = is_training
                
            y = self.sess.run(self.y, feed_dict=feed_dict)
            return y
    
    def reset_performance_stats(self):
        self.train_perf = {'loss': []}
        self.valid_perf = {'loss': [], 'psnr': [], 'ssim': []}        

    @property
    def name(self):
        if self.label is None:
            return '{}'.format(type(self).__name__)
        else:
            return '{}{}'.format(type(self).__name__, self.label)
    
    def count_parameters(self):
        return np.sum([np.prod(tv.shape.as_list()) for tv in self.parameters])
            
    def summary(self):
        return '{} model [{:,} params]'.format(type(self).__name__, self.count_parameters())
    

class UNet(NIPModel):
    """
    The UNet model, adapted from https://github.com/cchen156/Learning-to-See-in-the-Dark
    """
        
    def construct_model(self):
        with self.graph.as_default():            
            conv1 = slim.conv2d(self.x, 32, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv1_1'.format(self.label))
            conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv1_2'.format(self.label))
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='unet{}/max_pool_1'.format(self.label))

            conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv2_1'.format(self.label))
            conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv2_2'.format(self.label))
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='unet{}/max_pool_2'.format(self.label))

            conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv3_1'.format(self.label))
            conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv3_2'.format(self.label))
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='unet{}/max_pool_3'.format(self.label))

            conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv4_1'.format(self.label))
            conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv4_2'.format(self.label))
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME', scope='unet{}/max_pool_4'.format(self.label))

            conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv5_1'.format(self.label))
            conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv5_2'.format(self.label))

            up6 = upsample_and_concat(conv5, conv4, 256, 512, name='weights', scope='unet{}/upsample_1'.format(self.label))
            conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv6_1'.format(self.label))
            conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv6_2'.format(self.label))

            up7 = upsample_and_concat(conv6, conv3, 128, 256, name='weights', scope='unet{}/upsample_2'.format(self.label))
            conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv7_1'.format(self.label))
            conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv7_2'.format(self.label))

            up8 = upsample_and_concat(conv7, conv2, 64, 128, name='weights', scope='unet{}/upsample_3'.format(self.label))
            conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv8_1'.format(self.label))
            conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv8_2'.format(self.label))

            up9 = upsample_and_concat(conv8, conv1, 32, 64, name='weights', scope='unet{}/upsample_4'.format(self.label))
            conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv9_1'.format(self.label))
            conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='unet{}/conv9_2'.format(self.label))

            conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='unet{}/conv10'.format(self.label))

            with tf.name_scope('unet{}'.format(self.label)):
                self.yy = tf.depth_to_space(conv10, 2)
            self.y = tf.clip_by_value(self.yy, 0, 1, name='unet{}/y'.format(self.label))            


class INet(NIPModel):
    """
    A neural pipeline which replicates the steps of a standard imaging pipeline.
    """
    
    def construct_model(self, random_init=False, kernel=5, trainable_upsampling=False, cfa_pattern='gbrg'):
        self.trainable_upsampling = trainable_upsampling
        self.cfa_pattern = cfa_pattern

        with self.graph.as_default():
            with tf.variable_scope('inet{}'.format(self.label)):

                # Initialize the upsampling kernel
                upk = upsampling_kernel(cfa_pattern)

                if random_init:
                    # upk = np.random.normal(0, 0.1, (4, 12))
                    dmf = np.random.normal(0, 0.1, (kernel, kernel, 3, 3))
                    gamma_d1k = np.random.normal(0, 0.1, (3, 12))
                    gamma_d1b = np.zeros((12, ))
                    gamma_d2k = np.random.normal(0, 0.1, (12, 3))
                    gamma_d2b = np.zeros((3,))
                    srgbk = np.eye(3)
                else:    
                    # Prepare demosaicing kernels (bilinear)
                    dmf = bilin_kernel(kernel)

                    # Prepare gamma correction kernels (obtained from a pre-trained toy model)
                    gamma_d1k, gamma_d1b, gamma_d2k, gamma_d2b = gamma_kernels()

                    # Example sRGB conversion table
                    srgbk = np.array([[ 1.82691061, -0.65497452, -0.17193617],
                                      [-0.00683982,  1.33216381, -0.32532394],
                                      [ 0.06269717, -0.40055895,  1.33786178]]).transpose()

                # Up-sample the input back the full resolution
                with tf.variable_scope('upsampling'):
                    h12 = tf.layers.conv2d(self.x, 12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False, activation=None, name='conv_h12'.format(self.label), trainable=trainable_upsampling)

                # Demosaicing
                with tf.variable_scope('demosaicing'):
                    pad = (kernel - 1) // 2
                    bayer = tf.depth_to_space(h12, 2)
                    bayer = tf.pad(bayer, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
                    rgb = tf.layers.conv2d(bayer, 3, kernel, kernel_initializer=tf.constant_initializer(dmf), use_bias=False, activation=None, name='conv_demo'.format(self.label), padding='VALID')

                # Color space conversion
                with tf.variable_scope('rgb2sRGB'):
                    srgb = tf.layers.conv2d(rgb, 3, 1, kernel_initializer=tf.constant_initializer(srgbk), use_bias=False, activation=None, name='conv_sRGB'.format(self.label))

                # Gamma correction
                with tf.variable_scope('gamma'):
                    rgb_g0 = tf.layers.conv2d(srgb, 12, 1, kernel_initializer=tf.constant_initializer(gamma_d1k), bias_initializer=tf.constant_initializer(gamma_d1b), use_bias=True, activation=tf.nn.tanh, name='conv_encode'.format(self.label))
                    self.yy = tf.layers.conv2d(rgb_g0, 3, 1, kernel_initializer=tf.constant_initializer(gamma_d2k), bias_initializer=tf.constant_initializer(gamma_d2b), use_bias=True, activation=None, name='conv_decode'.format(self.label))
            
            self.y = tf.clip_by_value(self.yy, 0, 1, name='inet{}/y'.format(self.label))

    def init(self):
        # TODO That's a fairly ugly way to do it - need to find a better solution
        super().init()
        if not self.trainable_upsampling:
            with self.graph.as_default():
                with tf.variable_scope('inet{}/upsampling/conv_h12'.format(self.label), reuse=True):
                    self.sess.run(tf.variables_initializer([tf.get_variable('kernel')]))

    def load_model(self, camera_name, out_directory_root):
        if not self.trainable_upsampling:
            self.init()
        super().load_model(camera_name, out_directory_root)

            
class DNet(NIPModel):
    """
    Neural imaging pipeline adapted from a joint demosaicing-&-denoising model:
    Gharbi, Michaël, et al. "Deep joint demosaicking and denoising." ACM Transactions on Graphics (TOG) 35.6 (2016): 191.
    """

    def construct_model(self, n_layers=15, kernel=3, n_features=64):

        with self.graph.as_default():
                        
            with tf.name_scope('{}'.format(self.model_name)):
                self.is_training = tf.get_variable('is_training'.format(self.label), shape=(), dtype=tf.bool, initializer=tf.constant_initializer(True))
                k_initializer = tf.variance_scaling_initializer

                # Initialize the upsampling kernel
                upk = upsampling_kernel()

                # Padding size
                pad = (kernel - 1) // 2

                # Convolutions on the sub-sampled input tensor
                deep_x = self.x
                for r in range(n_layers):
                    deep_y = tf.layers.conv2d(deep_x, 12 if r == n_layers - 1 else n_features, kernel, use_bias=False, activation=None, name='{}/conv{}'.format(self.model_name, r), padding='VALID', kernel_initializer=k_initializer) #
                    print('CNN layer out: {}'.format(deep_y.shape))
                    deep_y = tf.layers.batch_normalization(deep_y, name='{}/bn{}'.format(self.model_name, r), training=self.is_training)
                    deep_y = tf.nn.relu(deep_y, name='{}/conv{}/Relu'.format(self.model_name, r))
                    deep_x = tf.pad(deep_y, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')

                # Upsample the input
                h12 = tf.layers.conv2d(self.x, 12, 1, kernel_initializer=tf.constant_initializer(upk), use_bias=False, activation=None, name='{}/conv_h12'.format(self.model_name), trainable=False)
                bayer = tf.depth_to_space(h12, 2, name="{}/upscaled_bayer".format(self.model_name))

                # Upscale the conv. features and concatenate with the input RGB channels
                features = tf.depth_to_space(deep_x, 2, name='{}/upscaled_features'.format(self.model_name))
                bayer_features = tf.concat((features, bayer), axis=3)            

                print('Final deep X: {}'.format(deep_x.shape))
                print('Bayer shape: {}'.format(bayer.shape))
                print('Features shape: {}'.format(features.shape))
                print('Concat shape: {}'.format(bayer_features.shape))

                # Project the concatenated 6-D features (R G B bayer from input + 3 channels from convolutions)
                pu = tf.layers.conv2d(bayer_features, n_features, kernel, kernel_initializer=k_initializer, use_bias=True, activation=tf.nn.relu, name='{}/conv_postupscale'.format(self.model_name), padding='VALID', bias_initializer=tf.zeros_initializer)

                print('Post upscale: {}'.format(pu.shape))

                # Final 1x1 conv to project each 64-D feature vector into the RGB colorspace
                pu = tf.pad(pu, tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]]), 'REFLECT')
                rgb = tf.layers.conv2d(pu, 3, 1, kernel_initializer=tf.ones_initializer, use_bias=False, activation=None, name='{}/conv_final'.format(self.model_name), padding='VALID')            

                print('RGB affine: {}'.format(rgb.shape))

                self.yy = rgb
                print('Y: {}'.format(self.yy.shape))

            self.y = tf.clip_by_value(self.yy, 0, 1, name='{}/y'.format(self.model_name))
