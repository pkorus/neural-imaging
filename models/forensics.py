# -*- coding: utf-8 -*-
"""
Provides image forensics models. See docs for the FAN class for details.
"""
import tensorflow as tf

from models.tfmodel import TFModel
from models.layers import ConstrainedConv2D
from helpers import paramspec, tf_helpers


class FAN(TFModel):
    """
    A forensic analysis network with a constrained residual filter. The model architecture is based on:

    [1] Bayar & Stamm, Constrained convolutional neural networks: A new approach towards general purpose image 
        manipulation detection. IEEE Transactions on Information Forensics and Security, 13 (11), 2018
    
    # Successive layers:

    1. A constrained conv layer (learned residual filter)
    2. N x standard conv layers
    3. A 1x1 conv layer
    4. GAP / flatten for feature extraction
    5. 2 hidden fully connected layers
    6. Output layer with K classes
    """

    def __init__(self, n_classes, patch_size=None, n_filters=32, n_fscale=2, n_convolutions=4, kernel=5, dropout=0.0, use_gap=True, n_dense=0, activation='leaky_relu'):
        """
        Creates a forensic analysis network (see class docstring for details).

        :param n_classes: the number of output classes
        :param patch_size: input patch size
        :param n_filters: number of output features for the first conv layer
        :param n_fscale: multiplier for the number of output features in successive conv layers
        :param n_convolutions: the number of standard conv layers
        :param kernel: conv kernel size
        :param dropout: dropout rate for fully connected layers
        :param use_gap: whether to use a GAP or to reshape the final conv tensor
        :param activation: activation function (see helpers.tf_helpers.activation_mapping for available activations)
        """
        super().__init__()

        # Set-up and validate hyper-parameters            
        self._h = paramspec.ParamSpec({
            'n_classes': (7, int, (2, 256)),
            'n_filters': (32, int, (4, 128)),
            'n_fscale': (2, float, (0.25, 4)),
            'n_convolutions': (4, int, (1, 32)),
            'kernel': (5, int, (3, 11)),
            'dropout': (0, float, (0, 1)),
            'use_gap': (False, bool, None),
            'n_dense': (2, int, (0, 16)),
            'activation': ('leaky_relu', str, set(tf_helpers.activation_mapping.keys()))
        })
        params = locals()
        self._h.update(**{k: params[k] for k in self._h.keys()})
        activation = tf_helpers.activation_mapping[self._h.activation]

        # Setup network input
        self.x = tf.keras.Input(dtype=tf.float32, shape=(patch_size, patch_size, 3))
        
        # Constrained convolution with a learned residual filter
        net = ConstrainedConv2D()(self.x)

        # Standard convolutional layers
        for _ in range(self._h.n_convolutions):
            net = tf.keras.layers.Conv2D(n_filters, [self._h.kernel, self._h.kernel], padding='same', activation=activation)(net)
            net = tf.keras.layers.MaxPool2D([2, 2])(net)
            n_filters = int(n_filters * self._h.n_fscale)

        n_filters = n_filters // n_fscale

        # Final 1 x 1 convolution
        net = tf.keras.layers.Conv2D(int(n_filters), [1, 1], activation=activation)(net)

        # GAP / Feature formation
        if use_gap:
            net = tf.keras.layers.GlobalAveragePooling2D()(net)
        else:
            net = tf.keras.layers.Flatten()(net)

        # Fully-connected classifier
        for _ in range(self._h.n_dense):
            n_filters = n_filters // n_fscale
            net = tf.keras.layers.Dense(n_filters, activation=activation)(net)
            if dropout > 0: net = tf.keras.layers.Dropout(dropout)(net)
        
        self.y = tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.softmax)(net)

        self._model = tf.keras.Model(inputs=self.x, outputs=self.y)        
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
            'accuracy': {'validation': []},
            'confusion': [],
        }

    def process(self, batch_x, training=False):
        """ Returns class probabilities for an image batch (NHWC:rgb). """
        return self._model(batch_x, training)

    def process_and_decide(self, batch_x, with_confidence=False):
        """ Returns the predicted class (and optionally its confidence) for an image batch (NHWC:rgb).  """
        probs = self._model(batch_x)

        if with_confidence:
            return probs.numpy().argmax(axis=1), probs.numpy().max(axis=1)
        else:
            return probs.numpy().argmax(axis=1)
        
    def training_step(self, batch_x, target_labels, learning_rate=None):
        """ Make a single training step and return the current loss. (Use class numbers for target labels.) """
        with tf.GradientTape() as tape:
            class_probabilities = self._model(batch_x)
            loss = self.loss(target_labels, class_probabilities)

        if learning_rate is not None: self.optimizer.lr.assign(learning_rate)
        grads = tape.gradient(loss, self._model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self._model.trainable_weights))
        return loss

    def summary(self):
        return '{kernel}x{kernel} CNN: 1+{conv}+1 conv layers {gap}+ {fc} fc layers [{params:,} parameters]'.format(
            kernel=self._h.kernel, 
            conv=self._h.n_convolutions, 
            fc=self._h.n_dense,
            gap='+ (GAP) ' if self._h.use_gap else '',
            params=self.count_parameters())
