import os
import tensorflow as tf
import numpy as np

class TFModel(object):

    def __init__(self, sess, graph, label, **kwargs):        
        self.graph = tf.Graph() if graph is None else graph
        self.sess = tf.Session(graph=self.graph) if sess is None else sess
        self.label = '_'+label if label is not None else ''
        self.model_name = '{}{}'.format(type(self).__name__.lower(), self.label)
        self.is_initialized = False
        self._saver = None
        self._summary_writer = None
        self.reset_performance_stats()        
        
    
    def reset_performance_stats(self):
        self.train_perf = {'loss': []}
        self.valid_perf = {'loss': []}

    def init(self):
        with self.graph.as_default():
            self.sess.run(tf.variables_initializer(self.parameters))
            self.sess.run(tf.variables_initializer(self.adam.variables()))
            self.is_initialized = True
            self._summary_writer = None
            self._saver = None
            self.reset_performance_stats()
    
#     def training_step(self, batch_x, batch_y, learning_rate, **kwargs):
#         raise NotImplementedError()
            
    @property
    def parameters(self):
        with self.graph.as_default():
            class_name = type(self).__name__.lower()
            trainable = tf.trainable_variables()
            return [tv for tv in tf.global_variables() if tv.name.startswith('{}{}/'.format(class_name, self.label)) and (tv in trainable or tv.name.endswith('moving_mean:0') or tv.name.endswith('moving_variance:0'))]
    
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
        class_name = type(self).__name__
        return '{}{} model [{:,} parameters]'.format(class_name, self.label, self.count_parameters())

    @property
    def saver(self):
#         class_name = type(self).__name__.lower()
        if not hasattr(self, '_saver') or self._saver is None:
            with self.graph.as_default():
                self._saver = tf.train.Saver(self.parameters, max_to_keep=0)
        return self._saver

    def save_model(self, dirname=None, epoch=0):
        class_name = type(self).__name__.lower()
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        
        with self.graph.as_default():
            self.saver.save(self.sess, os.path.join(dirname, class_name), global_step=epoch)

    def load_model(self, dirname):
        with self.graph.as_default():
            self.saver.restore(self.sess, tf.train.latest_checkpoint(dirname))
            
        self.is_initialized = True
        self.reset_performance_stats()
        
    @property
    def name(self):
        raise NotImplementedError()