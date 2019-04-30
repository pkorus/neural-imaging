import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

class DCN:
    """
    A forensic analysis network with the following architecture:

    1. A constrained conv layer (learned residual filter)
    2. N x standard conv layers
    3. A 1x1 conv layer
    4. GAP for feature extraction
    5. 2 hidden fully connected layers
    6. Output layer with K classes
    """

    def __init__(self, sess, graph, x=None, label=None, nip_input=None, patch_size=128, **kwargs):
        """
        Creates a forensic analysis network.

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param n_classes: the number of output classes
        :param x: input tensor
        :param nip_input: input to the NIP (if part of a larger model) or None
        :param n_filters: number of output features for the first conv layer
        :param n_fscale: multiplier for the number of output features in successive conv layers
        :param n_convolutions: the number of standard conv layers
        :param kernel: conv kernel size
        :param dropout: dropout rate for fully connected layers
        :param use_gap: whether to use a GAP or to reshape the final conv tensor
        """
        
        self.graph = tf.Graph() if graph is None else graph
        self.sess = tf.Session(graph=self.graph) if sess is None else sess
        self.label = '' if label is None else '_'+label
        self.patch_size = patch_size
        self.nip_input = nip_input
        
        with self.graph.as_default():
            
            # Setup inputs:
            # - if possible take external tensor as input, otherwise create a placeholder
            # - if external input is given (from a NIP model), remember the input to the NIP model to facilitate 
            #   convenient operation of the class (see helper methods 'process*')
            if x is None:
                x = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 3), name='x_dcn{}'.format(self.label))
                self.use_nip_input = False
            else:
                self.use_nip_input = True
            
            self.x = x
            self.construct_model(**kwargs)

            # Check if the model has set all expected attributes
            setup_status = {key: hasattr(self, key) for key in ['y', 'latent_pre', 'latent_post', 'latent_shape', 'n_latent']}
            if not all(setup_status.values()):
                raise NotImplementedError('The model construction function has failed to set-up some attributes: {}'.format([key for key, value in setup_status.items() if not value]))
                        
            with tf.name_scope('dcn{}/optimization'.format(self.label)):
                
                with tf.name_scope('entropy'):
                    
                    # Estimate entropy
                    values = tf.reshape(self.latent_post, (-1, 1))
                    bin_centers = tf.constant(np.arange(-127, 129), shape=(1, 256), dtype=tf.float32)
                    sigma = 1
                    weights = tf.exp(-sigma * tf.pow(values - bin_centers, 2))
                    print('Entropy weights', weights.shape)
                    self.weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

                    # Compute soft histogram
                    histogram = tf.clip_by_value(tf.reduce_mean(weights, axis=0), 1e-6, 1)
                    histogram = histogram / tf.reduce_sum(histogram)

                    self.entropy = - tf.reduce_sum(histogram * tf.log(histogram) / 0.6931) # 0.6931 - log(2)
                    self.histogram = histogram
                
                # Loss and SSIM
                self.ssim = tf.image.ssim(self.x, tf.clip_by_value(self.y, 0, 1), max_val=1)
                self.loss = tf.nn.l2_loss(self.x - self.y)
                
                # Optimization
                self.lr = tf.placeholder(tf.float32, name='dcn_learning_rate')
                self.adam = tf.train.AdamOptimizer(learning_rate=self.lr)
                self.opt = self.adam.minimize(self.loss, var_list=self.parameters)
                
        self.is_initialized = False
        self.reset_performance_stats()

    def construct_model(self):
        raise Error('Not implemented!')
        
    def reset_performance_stats(self):
        self.train_perf = {'loss': []}
        self.valid_perf = {'loss': []}

    def init(self):
        with self.graph.as_default():
            self.sess.run(tf.variables_initializer(self.parameters))
            self.sess.run(tf.variables_initializer(self.adam.variables()))
            for k, v in self.train_perf.items():
                v.clear()
            for k, v in self.valid_perf.items():
                v.clear()

            self.is_initialized = True
            self._summary_writer = None
            self.reset_performance_stats()

    def compress(self, batch_x):
        with self.graph.as_default():
            y = self.sess.run(self.latent_post, feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x
            })
            return y

    def compress_soft(self, batch_x):
        with self.graph.as_default():
            y = self.sess.run(self.latent_pre, feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x
            })
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
            
    def process(self, batch_x, dropout_keep_prob=1.0):
        """
        Returns the predicted class for an image batch. The input is fed to the NIP if the model is chained properly.
        """
        with self.graph.as_default():
            feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x
            }
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = dropout_keep_prob
                
            y = self.sess.run(self.y, feed_dict)
            return y.clip(0, 1)
    
    def process_direct(self, batch_x, dropout_keep_prob=1.0):
        """
        Returns the predicted class for an image batch. The input is always fed to the FAN model directly.
        """
        with self.graph.as_default():
            feed_dict = {
                self.x: batch_x
            }
            if hasattr(self, 'dropout'):
                feed_dict[self.dropout] = dropout_keep_prob
                
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
        
    @property
    def parameters(self):
        with self.graph.as_default():
            return [tv for tv in tf.trainable_variables() if tv.name.startswith('dcn/')]
    
    def get_summary_writer(self, dirname):
        if not hasattr(self, '_summary_writer') or self._summary_writer is None:
            
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            
            with self.graph.as_default():
                self._summary_writer = tf.summary.FileWriter(dirname, self.graph)
                
        return self._summary_writer
        
    def count_parameters(self):
        return np.sum([np.prod(tv.shape.as_list()) for tv in self.parameters])
    
    def count_parameters_breakdown(self):
        return OrderedDict([(tv.name, np.prod(tv.shape.as_list())) for tv in self.parameters])
    
    def summary(self):
        return 'dcn with {} conv layers and {}-D latent representation [{:,} parameters]'.format(self.n_layers, self.n_latent, self.count_parameters())

    @property
    def saver(self):
        if not hasattr(self, '_saver') or self._saver is None:
            with self.graph.as_default():
                self._saver = tf.train.Saver(self.parameters, max_to_keep=0)
        return self._saver

    def save_model(self, dirname, epoch):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        with self.graph.as_default():
            self.saver.save(self.sess, os.path.join(dirname, 'dcn'), global_step=epoch)

    def load_model(self, dirname):
        with self.graph.as_default():
            self.saver.restore(self.sess, tf.train.latest_checkpoint(dirname))
            
        self.is_initialized = True
        self.reset_performance_stats()
        
    def short_name(self):
        # If the latent representation 
        if hasattr(self, 'latent_shape'):
            dim_string = 'x'.join(str(x) for x in self.latent_shape[1:])
        elif hasattr(self, 'n_latent'):
            dim_string = '{}D'.format(self.n_latent)
        else:
            raise ValueError('The model does not report the latent space dimensionality.')
        
        return '{}-{}'.format(type(self).__name__, dim_string)