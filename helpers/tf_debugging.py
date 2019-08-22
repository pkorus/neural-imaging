import tensorflow as tf
import numpy as np
from collections import OrderedDict


def get_tr_gradients(loss):
    trainables = tf.trainable_variables()
    grads = tf.gradients(loss, trainables)
    mapping = OrderedDict()
    for k, v in zip(trainables, grads):
        mapping[k.name] = v
    return mapping


def get_op_gradients(graph, loss, prefix='distribution/', blacklist='Const,StopGradient,Reshape,Transpose'):
    blacklist = blacklist.split(',')

    grads_c = tf.gradients(loss, [x.outputs[0] for x in graph.get_operations() if x.name.startswith(prefix) and x.type not in blacklist])
    gradn_c = [x.name for x in graph.get_operations() if x.name.startswith(prefix) and x.type not in blacklist]

    mapping = OrderedDict()

    for g, n in zip(grads_c, gradn_c):
        if g is not None:
            mapping[n] = g

    return mapping


def summarize_nan_inf(label, tensor):
    nans = np.mean(np.isnan(tensor))
    infs = np.mean(np.isinf(tensor))
    nans_l = '-' if nans == 0 else '{:>10.3f}'.format(100*nans)
    infs_l = '-' if infs == 0 else '{:>10.3f}'.format(100*infs)
    print('{:>20s}{:>10s}{:>10s}'.format(label, nans_l, infs_l))


def print_grad_summary(grad_values, labels):
    for index, label in enumerate(labels):
        print('\n{:<70.70s}\t{:20s}\t{:>10s}\t{:>10s}'.format(label, 'Shape', 'Nans', 'Infs'))
        for k, v in grad_values[index].items():
            nans = np.mean(np.isnan(v))
            infs = np.mean(np.isinf(v))
            nans_l = '-' if nans == 0 else '{:>10.3f}'.format(100 * nans)
            infs_l = '-' if infs == 0 else '{:>10.3f}'.format(100 * infs)
            print('{:<70.70s}\t{:20s}\t{:>10s}\t{:>10s}'.format(k, str(v.shape), nans_l, infs_l))


def check_gradients(grad_values):
    for group in grad_values:
        for k, v in group.items():
            nans = np.mean(np.isnan(v))
            infs = np.mean(np.isinf(v))
            if nans > 0 or infs > 0:
                return False
    return True
