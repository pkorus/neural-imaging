import os
import sys

import tensorflow as tf
from six.moves import urllib


def lpips(img_a, img_b, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        input0: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].
        input1: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].

    Returns:
        The Learned Perceptual Image Patch Similarity (LPIPS) distance.

    Models:
        http://rail.eecs.berkeley.edu/models/lpips

    Adapted from:
        https://github.com/alexlee-gk/lpips-tensorflow

    Reference:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    """

    image0 = tf.placeholder(tf.float32)
    image1 = tf.placeholder(tf.float32)

    # flatten the leading dimensions
    batch_shape = tf.shape(image0)[:-3]
    input0 = tf.reshape(image0, tf.concat([[-1], tf.shape(image0)[-3:]], axis=0))
    input1 = tf.reshape(image1, tf.concat([[-1], tf.shape(image1)[-3:]], axis=0))

    # NHWC to NCHW
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])

    # normalize to [-1, 1]
    input0 = input0 * 2.0 - 1.0
    input1 = input1 * 2.0 - 1.0

    input0_name, input1_name = '0:0', '1:0'

    default_graph = tf.get_default_graph()
    cache_dir = os.path.expanduser('../data/lpips')

    # files to try. try a specific producer version, but fallback to the version-less version (latest).
    pb_fname = '%s_%s_v%s.pb' % (model, net, version)

    if not os.path.isfile(os.path.join(cache_dir, pb_fname)):
        raise ValueError('Model not available! {}'.format(pb_fname))

    with open(os.path.join(cache_dir, pb_fname), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, input_map={input0_name: input0, input1_name: input1})
        distance, = default_graph.get_operations()[-1].outputs

        # for op in default_graph.get_operations():
        #     print(op.name, op.outputs[0].shape)

    if distance.shape.ndims == 4:
        distance = tf.squeeze(distance, axis=[-3, -2, -1])

    # reshape the leading dimensions
    distance_t = tf.reshape(distance, batch_shape)

    with tf.Session() as session:
        return session.run(distance_t, feed_dict={image0: img_a, image1: img_b})

