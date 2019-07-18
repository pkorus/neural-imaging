import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import helpers.tf_helpers
from models.tfmodel import TFModel
from helpers import utils


class FAN(TFModel):
    """
    A forensic analysis network with the following architecture:

    1. A constrained conv layer (learned residual filter)
    2. N x standard conv layers
    3. A 1x1 conv layer
    4. GAP for feature extraction
    5. 2 hidden fully connected layers
    6. Output layer with K classes
    """

    def __init__(self, sess, graph, n_classes, x, label=None, nip_input=None, n_filters=32, n_fscale=2, n_convolutions=3, kernel=5, dropout=0.0, use_gap=True):
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
        super().__init__(sess, graph, label)

        with self.graph.as_default():
            
            # Setup inputs:
            # - if possible take external tensor as input, otherwise create a placeholder
            # - if external input is given (from a NIP model), remember the input to the NIP model to facilitate 
            #   convenient operation of the class (see helper methods 'process*')
            if x is None:
                x = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='x_fan')
                self.use_nip_input = False
            else:
                self.use_nip_input = True
                
            # Setup a GT placeholder
            y = tf.placeholder(tf.int32, shape=(None,), name='y_fan')
            
            with tf.name_scope('fan'):
                
                print('Building CNN for {} output classes'.format(n_classes))
                # Basic parameters
                activation = helpers.tf_helpers.lrelu
                filter_strength = 100

                # First residual filter
                with tf.name_scope('residual'):
                    f = np.array([[0, 0, 0, 0, 0], [0, -1, -2, -1, 0], [0, -2, 12, -2, 0], [0, -1, -2, -1, 0], [0, 0, 0, 0, 0]])        
                    rf = utils.repeat_2dfilter(f, 3)

                    # Mask for normalizing the residual filter                
                    tf_ind = tf.constant(utils.center_mask_2dfilter(5, 3), dtype=tf.float32)

                    # Allocate a TF variable for the filter coefficients
                    prefilter_kernel = tf.get_variable('fan/residual/conv_res/weights', shape=(5, 5, 3, 3), initializer=tf.constant_initializer(rf))

                    # Normalize the residual filter
                    nf = prefilter_kernel * (1 - tf_ind)
                    df = tf.tile(tf.reshape(tf.reduce_sum(nf, axis=(0,1,2)), [1, 1, 1, 3]), [5, 5, 3, 1])
                    nf = filter_strength * nf / df
                    nf = nf - filter_strength * tf_ind        

                    # Convolution with the residual filter
                    xp = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], 'SYMMETRIC')
                    net = tf.nn.conv2d(xp, nf, [1, 1, 1, 1], 'VALID')
                    print('{}x{} conv prefilter shape {}'.format(5, 5, net.shape))

            # Standard convolutional layers
            for conv_id in range(n_convolutions):
                net = slim.conv2d(net, n_filters, [kernel, kernel], rate=1, activation_fn=activation, scope='fan/conv{}'.format(conv_id+1), reuse=False)
                print('{}x{} conv {} shape {} + {} + 2x2 pool'.format(kernel, kernel, conv_id+1, net.shape, activation.__name__))
                net = slim.max_pool2d(net, [2, 2], scope='fan/maxpool{}'.format(conv_id+1))
                n_filters *= n_fscale

            # Final 1 x 1 convolution
            net = slim.conv2d(net, n_filters // n_fscale, [1, 1], rate=1, activation_fn=activation, scope='fan/conv{}'.format(conv_id+2), reuse=False)
            print('{}x{} conv {} shape {} + {}'.format(1, 1, conv_id+2, net.shape, activation.__name__))    

            # GAP / Feature formation
            print('Final conv shape', net.shape)
            with tf.name_scope('fan/'):
                if use_gap:
                    net = tf.reduce_mean(net, axis=(1, 2), name='gap')
                else:
                    net = tf.reshape(net, (-1, net.shape[1] * net.shape[2] * net.shape[3]))
                # Remember extracted features for future debugging
                self.features = net
                print('Feature shape', net.shape)

            # Fully-connected classifier
            net = slim.fully_connected(net, 512, activation_fn=activation, scope='fan/ff1', reuse=False)
            print('fc shape: {} + {}'.format(net.shape, activation.__name__))
            if dropout > 0:
                net = slim.dropout(net, dropout, scope='fan/dropout_ff1')
                print('{:.1f} dropout'.format(dropout))
            net = slim.fully_connected(net, 128, activation_fn=activation, scope='fan/ff2', reuse=False)
            print('fc shape: {} + {}'.format(net.shape, activation.__name__))
            if dropout > 0:
                net = slim.dropout(net, dropout, scope='fan/dropout_ff2')
                print('{:.1f} dropout'.format(dropout))

            # Output layer
            y_ = slim.fully_connected(net, n_classes, activation_fn=tf.nn.softmax, scope='fan/predictions', reuse=False)
            print('out shape: {} + {}'.format(y_.shape, tf.nn.softmax.__name__))

            with tf.name_scope('fan_optimization'):
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=y))
                lr = tf.placeholder(tf.float32, name='fan_learning_rate')
                adam = tf.train.AdamOptimizer(learning_rate=lr, name='adam')
                opt = adam.minimize(loss, name='fan_opt_all_models')
                opt_own = adam.minimize(loss, var_list=self.parameters, name='fan_opt_fan_only')
        
        self.x = x
        self.y = y
        self.y_ = y_
        self.loss = loss
        self.lr = lr
        self.adam = adam
        self.opt = opt
        self.opt_own = opt_own
        self.nip_input = nip_input
        self.n_classes = n_classes
        self.n_convolutions = n_convolutions
        self.kernel = kernel

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'accuracy': {'validation': []},
        }

    def process(self, batch_x):
        """
        Returns the predicted class for an image batch. The input is fed to the NIP if the model is chained properly.
        """
        with self.graph.as_default():
            y = self.sess.run(self.y_, feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x
            })
            return np.argmax(y, axis=1)

    def process_soft(self, batch_x):
        """
        Returns class probabilities for an image batch. The input is fed to the NIP if the model is chained properly.
        """
        with self.graph.as_default():
            y = self.sess.run(self.y_, feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x
            })
            return y    
    
    def process_direct(self, batch_x, with_confidence=False):
        """
        Returns the predicted class for an image batch. The input is always fed to the FAN model directly.
        """
        with self.graph.as_default():
            y = self.sess.run(self.y_, feed_dict={
                self.x: batch_x
            })
            return (np.argmax(y, axis=1), np.max(y, axis=1)) if with_confidence else np.argmax(y, axis=1)

    def process_with_loss(self, batch_x, batch_y):
        """
        Returns the predicted class and loss for an image batch.
        """
        with self.graph.as_default():
            y, loss_value = self.sess.run([self.y_, self.loss], feed_dict={
                self.x if not self.use_nip_input else self.nip_input: batch_x,
                self.y: batch_y
            })
            return np.argmax(y, axis=1), loss_value    
    
    def training_step(self, batch_x, batch_y, learning_rate):
        """
        Make a single training step and return current loss. Only the FAN model is updated.
        """
        with self.graph.as_default():
            _, loss = self.sess.run([self.opt_own, self.loss], feed_dict={
                    self.x if not self.use_nip_input else self.nip_input: batch_x,
                    self.y: batch_y,
                    self.lr: learning_rate
                    })
            return loss
    
    def training_step_all_models(self, batch_x, batch_y, learning_rate):
        """
        Make a single training step and return current loss. All relevant models are updated.
        """
        with self.graph.as_default():
            _, loss = self.sess.run([self.opt, self.loss], feed_dict={
                    self.x if not self.use_nip_input else self.nip_input: batch_x,
                    self.y: batch_y,
                    self.lr: learning_rate
                    })
            return loss

    def summary(self):
        return '{}x{} cnn with {}+1+1 conv layers + 2 fc layers [{:,} parameters]'.format(self.kernel, self.kernel, self.n_convolutions, self.count_parameters())
