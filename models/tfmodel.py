"""
Provides the base class for all toolbox models. See docs for `TFModel` for more details.
"""
import os
import tensorflow as tf
import numpy as np
import json

from pathlib import Path

from helpers import utils

from loguru import logger


def restore(dir_name, module, key=None, patch_size=None, restore_perf=False, fetch_stats=False):
    """
    Utility function to restore pre-trained models from a training directory. 

    :param dir_name: directory with a trained model (*.json + checkpoint data)
    :param module: Python module where classes should be looked up
    :param key: JSON key which describes which model to look up in the training log
    :param patch_size: input patch size (scalar)
    :param restore_perf: also loads training/validation metrics
    :param fetch_stats: return a tuple (model, training_stats)
    """
    training_log_path = None

    if dir_name is None:
        raise ValueError('dcn directory cannot be None')

    if not os.path.exists(dir_name):
        # If not explicit directory, check for presets
        logger.info('config/presets/{}.json'.format(module.__name__.split('.')[-1]))
        if os.path.isfile('config/presets/{}.json'.format(module.__name__.split('.')[-1])):
            with open('config/presets/{}.json'.format(module.__name__.split('.')[-1])) as f:
                presets = json.load(f)
            if dir_name in presets:
                logger.info('Found {} in presets: {}'.format(dir_name, presets[dir_name]))
                dir_name = presets[dir_name]
            else:
                raise ValueError('Directory {} does not exist & key not found in presets (config/presets/*)!'.format(dir_name))
        else:
            raise ValueError('Directory {} does not exist (presets not available)!'.format(dir_name))

    for filename in Path(dir_name).glob('**/*.json'):
        training_log_path = str(filename)

    if training_log_path is None:
        raise FileNotFoundError('Could not find a training log (JSON file) in {}'.format(dir_name))

    with open(training_log_path) as f:
        training_log = json.load(f)
    
    if key is not None:
        training_log = training_log[key]

    parameters = training_log['args']
    parameters['patch_size'] = patch_size

    # TODO JSON Does not allow to store tuples, so they are stored as string
    for key, value in parameters.items():
        if isinstance(value, str) and value[0] == '(' and value[-1] == ')':
            parameters[key] = eval(value)

    model = getattr(module, training_log['model'])(**parameters)
    model.load_model(dir_name)
    logger.info('Restored model: {} <- {}'.format(model.model_code, training_log_path))

    if restore_perf:
        model.performance = training_log['performance']

    if fetch_stats:
        stats = {}
        for k, v in model.performance.items():
            if 'validation' in v and len(v['validation']) > 0:
                stats[k] = np.round(v['validation'][-1], 3)
            elif 'training' in v and len(v['training']) > 0:
                stats[k] = np.round(v['training'][-1], 3)

        return model, stats
    else:
        return model


class TFModel(object):
    """
    Abstract class to represent framework components. Provides common functionality to keep
    performance statistics, help with model loading/saving/migration, access and count parameters, 
    hyper-parameters, etc. For most use-cases, see specific sub-classes: e.g, NIPModel for camera 
    ISPs, or DCN for learned compression.

    # Working with hyper-parameters
    The framework provides the 'ParamSpec' class to help with hyper-parameter definitions, validation
    and storage. See documentation of that class for details, and existing TFModel sub-classes for 
    more examples.

    # Accessing model parameters
    - parameters - list of all trainable parameters in the model (useful for loading/saving/counting parameters)
    - variables  - list of all variables in the model (useful for initialization)

    # Usage of model strings in the framework:
    - summary                 - a human-readable summary of the model (name + rudimentary layer specs + parameter count)
    - model_code              - represents a concise, coded summary of the models hyper parameters
    - class_name              - convenience method to access class name
    - scoped_name             - class name (lower case) [+ postfix label] (e.g., unet / unet_a / fan)
                                used as a directory name for storing models
    """

    def __init__(self, **kwargs):
        self._model = None
        self.reset_performance_stats()        

    @staticmethod
    def _reset_performance(metrics):
        return {k: {'training': [], 'validation': []} for k in metrics}

    def reset_performance_stats(self):
        self.performance = self._reset_performance(['loss'])

    def log_metric(self, metric, scope, value, raw=False):
        if not raw:
            if utils.is_number(value):
                value = float(value)
            else:
                value = float(np.mean(value))

        self.performance[metric][scope].append(value)

    def pop_metric(self, metric, scope):
        return self.performance[metric][scope][-1]

    @property
    def parameters(self):
        return self._model.trainable_weights
    
    @property
    def variables(self):
        return self._model.variables
        
    def count_parameters(self):
        return np.sum([np.prod(tv.shape.as_list()) for tv in self.parameters])
    
    def count_parameters_breakdown(self):
        import pandas as pd
        total = self.count_parameters()
        data = [(tv.name, tv.shape, np.prod(tv.shape.as_list()), round(100 * np.prod(tv.shape.as_list()) / total, 1)) for tv in self.parameters]
        return pd.DataFrame(data, columns=['name', 'shape', 'parameters', 'total'])        

    def save_model(self, dirname, epoch=0, save_args=False, quiet=False):
        if not dirname.endswith(self.scoped_name):
            dirname = os.path.join(dirname, self.scoped_name)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if not quiet:
            logger.info(f'> {self.class_name} --> {os.path.join(dirname, self.class_name.lower())} {"JSON" if save_args else ""}')
        self._model.save_weights(os.path.join(dirname, f'{self.class_name.lower()}.h5'), save_format='h5')

        if save_args:
            with open(os.path.join(dirname, f'{self.class_name.lower()}.json'), 'w') as f:
                json.dump({
                    'model': self.class_name,
                    'args': self.get_hyperparameters()
                }, f, indent=4)

    def load_model(self, dirname, quiet=False):
        if not dirname.endswith(self.scoped_name):
            dirname = os.path.join(dirname, self.scoped_name)

        filename = os.path.join(dirname, f'{self.class_name.lower()}.h5')

        # If the h5 model does not exist, try the TF snapshot
        if not os.path.isfile(filename):
            filename = os.path.join(dirname, self.class_name.lower())

        if not quiet:
            logger.info(f'> {self.class_name} <-- {filename}')

        self._model.load_weights(filename)
        self.reset_performance_stats()

    def migrate_model(self, dirname, mapping=None, verbose=False):
        """
        Migrate a pre-trained model from a TF checkpoint. Example cases when this arises include
        changed TF version or changed variable names. The function loads specific variables from
        the checkpoint and uses their values for new weights. The mapping is defined in the
        'mapping' dictionary. The new model can later be saved using 'save_model'.

        Hint: It may be useful to use tf.keras.backend.clear_session() to make sure variable 
        names are not changing during the migration.

        :param dirname: directory with a saved TF checkpoint
        :param mapping: dict {'new name' : 'old name'}
        :param verbose: self explanatory
        """
        if not dirname.endswith(self.scoped_name):
            dirname = os.path.join(dirname, self.scoped_name)

        if verbose:
            logger.info('# Variables found in the checkpoint: {}'.format(dirname))
            for i, (var_name, _) in enumerate(tf.train.list_variables(dirname)):
                var = tf.train.load_variable(dirname, var_name)
                if hasattr(var, 'shape'):
                    logger.info('{0:3d}.  {1:70s} -> tensor {2.shape}'.format(i, var_name, var))
                else:
                    logger.info('{0:3d}.  {1:70s} -> {2}'.format(i, var_name, type(var)))
            logger.info('\n# Model variables: {}'.format(self.class_name))
            for i, var in enumerate(self._model.trainable_variables):
                logger.info('{0:3d}.  {1.name:70s} -> tensor {1.shape}'.format(i, var))

        if mapping is not None:
            for var in self._model.trainable_variables:
                var_name = var.name.replace(':0', '')
                if var_name not in mapping:
                    logger.warning('mapping for {} = {} not found'.format(var.name, var_name))
                    continue
                var_value = tf.train.load_variable(dirname, mapping[var_name])
                logger.info('{} = {} {} <- {} {}'.format(var.name, var_name, var.shape, mapping[var_name], var_value.shape))
                var.assign(var_value)
        
        self.reset_performance_stats()

    @property
    def class_name(self):
        return type(self).__name__

    def summary(self):
        return '{} model [{:,.0f} parameters]'.format(self.class_name, self.count_parameters())

    def summary_compact(self):
        return '{}'.format(self.class_name)

    @property
    def model_code(self):
        raise NotImplementedError()

    @property
    def scoped_name(self):
        return '{}'.format(type(self).__name__.lower())

    def get_hyperparameters(self):
        if hasattr(self, '_h'):
            return self._h.to_json()
        else:
            return None

    def __repr__(self):
        try:
            extra_params = utils.join_args(self._h.changed_params())
        except:
            extra_params = ''
        return f'{self.class_name}({extra_params})'

    def _has_attributes(self, attrs, message='Expected attributes not found: {}'):
        setup_status = {key: hasattr(self, key) for key in attrs}
        if not all(setup_status.values()):
            raise NotImplementedError(message.format([key for key, value in setup_status.items() if not value]))

    @classmethod
    def restore(cls, dir_name, *, key=None, patch_size=None):

        candidates = list(Path(dir_name).glob('**/*.json'))
        training_log_path = str(candidates[0]) if candidates else None

        if training_log_path is None or not os.path.isfile(training_log_path):
            raise FileNotFoundError('Could not find a training log (JSON file) in {}'.format(dir_name))

        with open(training_log_path) as f:
            training_log = json.load(f)

        if key is not None:
            training_log = training_log[key]

        parameters = training_log['args']
        if patch_size is not None: parameters['patch_size'] = patch_size

        # JSON does not allow to store tuples, so they are stored as string
        for key, value in parameters.items():
            if isinstance(value, str) and value[0] == '(' and value[-1] == ')':
                parameters[key] = eval(value)

        instance = cls(**parameters)
        instance.load_model(dir_name)
        
        return instance

    def process(self, x, training=False):
        return self._model(x, training)

    def deploy_model(self, dirname):
        # TODO Need to implement model deployment - need to set input shape & self._model.save(dirname)
        raise NotImplementedError()
