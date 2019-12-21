import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from collections import OrderedDict


class TFModel(object):
    """
    Class to represent TF models with model saving / loading capabilities.

    # Accessing model parameters
    - parameters - list of all trainable parameters in the model (useful for loading/saving/counting parameters)
    - variables  - list of all variables in the model (useful for initialization)

    # Usage of model strings in the framework:
    - summary                 - a human-readable summary of the model (name + rudimentary layer specs + parameter count)
    - model_code              - represents a concise, coded summary of the models hyper parameters
    - class_name              - convenience method to access class name
    - scoped_name             - class name (lower case) [+ postfix label] (e.g., unet / unet_a / fan)
                                used as a prefix for TF variables & as a directory name for storing models
    """

    def __init__(self, sess, graph, label, **kwargs):        
        self.graph = tf.Graph() if graph is None else graph
        self.sess = tf.Session(graph=self.graph) if sess is None else sess
        self._label = '_'+label if label is not None else ''
        self.is_initialized = False
        self._saver = None
        self._summary_writer = None
        self.reset_performance_stats()        

    def reset_performance_stats(self):
        self.performance = {
            'loss': {'training': [], 'validation': []},
        }

    def init(self):
        with self.graph.as_default():
            self.sess.run(tf.variables_initializer(self.variables))
            self.is_initialized = True
            self._summary_writer = None
            self._saver = None
            self.reset_performance_stats()

    @property
    def parameters(self):
        with self.graph.as_default():
            # TODO the current implementation needs to manually add population statistics from BN layers since these
            # TODO variables are updated manually and are not reported as 'trainable'
            trainable = tf.trainable_variables()
            return [tv for tv in tf.global_variables() if tv.name.startswith('{}/'.format(self.scoped_name)) and (tv in trainable or tv.name.endswith('moving_mean:0') or tv.name.endswith('moving_variance:0'))]
    
    @property
    def variables(self):
        """ List all variables related to the model - regardless of whether they are trainable """
        with self.graph.as_default():
            return [tv for tv in tf.global_variables() if tv.name.startswith('{}/'.format(self.scoped_name))]

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

    @property
    def saver(self):
        if not hasattr(self, '_saver') or self._saver is None:
            with self.graph.as_default():
                self._saver = tf.train.Saver(self.parameters, max_to_keep=5)
        return self._saver

    def save_model(self, dirname, epoch=0):

        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        with self.graph.as_default():
            self.saver.save(self.sess, os.path.join(dirname, self.class_name.lower()), global_step=epoch)

    def load_model(self, dirname):

        self.init()

        # Try to load the model from the given directory
        latest_checkpoint = tf.train.latest_checkpoint(dirname)

        # If no model available, append current model's scoped name
        if latest_checkpoint is None:
            dirname = os.path.join(dirname, self.scoped_name)
            latest_checkpoint = tf.train.latest_checkpoint(dirname)

        if latest_checkpoint is None:
            raise RuntimeError('Model checkpoint not found at {}'.format(dirname))

        with self.graph.as_default():
            # Use the slim package to load the checkpoint - this gives a chance to ignore missing variables
            init_assign_op, init_feed_dict = slim.assign_from_checkpoint(latest_checkpoint, self.parameters, ignore_missing_vars=True)
            self.sess.run(init_assign_op, feed_dict=init_feed_dict)

        self.is_initialized = True
        self.reset_performance_stats()

    @property
    def class_name(self):
        return type(self).__name__

    def summary(self):
        return '{} model [{:,} parameters]'.format(self.class_name, self.count_parameters())

    @property
    def model_code(self):
        raise NotImplementedError()

    @property
    def scoped_name(self):
        return '{}{}'.format(type(self).__name__.lower(), self._label)
