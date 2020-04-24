# -*- coding: utf-8 -*-
"""
Various helper functions for Tensorflow.

Example functionality:
----------------------
- various image losses
- correlation coefficients
- image manipulations
- image quantization and clipping
- visualize structure of Keras Models
- differentiable entropy approximation

"""
import os
import tensorflow as tf
import numpy as np

from helpers.kernels import gkern, repeat_2dfilter
from IPython.display import display, HTML

activation_mapping = {
    'leaky_relu' : tf.keras.layers.LeakyReLU(alpha=0.2),
    'relu': tf.keras.activations.relu,
    'tanh': tf.keras.activations.tanh,
    'sigmoid': tf.keras.activations.sigmoid,
    'softsign': tf.keras.activations.softsign
}


def mse(a, b):
    return tf.reduce_mean(tf.math.pow(255 * a - 255 * b, 2.0))


def mae(a, b):
    return tf.reduce_mean(tf.math.abs(255 * a - 255 * b))


def ssim_loss(a, b):
    return tf.reduce_mean(255 * (1 - tf.image.ssim(a, b, 1.0)))


def msssim_loss(a, b):
    return tf.reduce_mean(255 * (1 - tf.image.ssim_multiscale(a, b, 1.0)))


def corr(a, b):
    a = (a - tf.reduce_mean(a, axis=[1, 2, 3], keepdims=True)) / (1e-9 + tf.math.reduce_std(a, axis=[1, 2, 3], keepdims=True))
    b = (b - tf.reduce_mean(b, axis=[1, 2, 3], keepdims=True)) / (1e-9 + tf.math.reduce_std(b, axis=[1, 2, 3], keepdims=True))
    c = tf.reduce_mean(a * b, axis=[1, 2, 3])
    return c


def corrcoeff(a, b):
    a = (a - tf.reduce_mean(a)) / (1e-9 + tf.math.reduce_std(a))
    b = (b - tf.reduce_mean(b)) / (1e-9 + tf.math.reduce_std(b))
    c = tf.reduce_mean(a * b)
    return c.numpy()


def rsquared(a, b):
    from sklearn.metrics import r2_score 
    a = (a - tf.reduce_mean(a)) / (1e-9 + tf.math.reduce_std(a))
    b = (b - tf.reduce_mean(b)) / (1e-9 + tf.math.reduce_std(b))
    return r2_score(a, b)


def manipulation_resample(x, factor=50, method='bilinear'):

    if 0 < factor <= 1:
        factor = 100 * factor

    output_shape = [tf.shape(x)[1] * int(factor) // 100, tf.shape(x)[1] * int(factor) // 100]

    im_res = tf.image.resize(x, output_shape, method=method)
    return tf.image.resize(im_res, [tf.shape(x)[1], tf.shape(x)[1]], method)


def manipulation_awgn(x, strength=0.025):
    im_awgn = x + strength * tf.random.normal(tf.shape(x))
    im_awgn = soft_quantization(im_awgn)
    return tf.clip_by_value(im_awgn, 0, 1)


def manipulation_gamma(x, strength=2.0):
    im_gamma = tf.pow(x, strength)
    im_gamma = soft_quantization(im_gamma)
    return tf.pow(tf.clip_by_value(im_gamma, 1.0/255, 1), 1/strength)


def manipulation_median(x, kernel=3):    
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(kernel, 1)

    xp = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')
    patches = tf.image.extract_patches(xp, [1, kernel, kernel, 1], [1, 1, 1, 1], 4*[1], 'VALID')
    patches = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2], tf.shape(patches)[3]//3, 3])
    patches = tf.transpose(patches, [0, 1, 2, 4, 3])

    area = kernel ** 2
    floor = (area + 1) // 2
    ceil = area // 2 + 1

    top = tf.nn.top_k(patches, k=ceil).values
    # The area will always be odd if kernel is odd
    median = top[:, :, :, :, floor - 1]

    return median


def manipulation_gaussian(x, kernel, std, skip_clip=False):
    kernel = int(kernel)
    gk = gkern(kernel, std)
    gfilter = np.zeros((kernel, kernel, 3, 3))
    for r in range(3):
        gfilter[:, :, r, r] = gk
    gkk = tf.constant(gfilter, tf.float32)
    xp = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')
    y = tf.nn.conv2d(xp, gkk, [1, 1, 1, 1], 'VALID')
    if skip_clip:
        return y
    else:
        return tf.clip_by_value(y, 0, 1)

def residual(x, hsv=False):
    with tf.name_scope('residual_filter'):

        # Prepare the sharpening filter
        gk = np.array([[-0.0833, -0.1667, -0.0833], [-0.1667, 1, -0.1667], [-0.0833, -0.1667, -0.0833]])

        if gk is None or gk.ndim != 2 or gk.shape[0] != gk.shape[1]:
            raise ValueError('Invalid filter! {}'.format(gk))

        kernel = gk.shape[0]
        gfilter = repeat_2dfilter(gk, 3)
        if hsv:
            gfilter[:, :, 1:2, 1:2] = 0
            gfilter[2, 2, 1:2, 1:2] = 1

        gkk = tf.constant(gfilter, tf.float32)

        y = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')

        if hsv:
            y = tf.image.rgb_to_hsv(y)

        y = tf.nn.conv2d(y, gkk, [1, 1, 1, 1], 'VALID')

        if hsv:
            y = tf.image.hsv_to_rgb(y)

        return y

def manipulation_sharpen(x, strength=1, hsv=True):
    # Prepare the sharpening filter
    gk = np.array([[-0.0833, -0.1667, -0.0833], [-0.1667, 0, -0.1667], [-0.0833, -0.1667, -0.0833]])
    gk = strength * gk / np.abs(gk.sum())
    gk[1, 1] = strength + 1

    if gk is None or gk.ndim != 2 or gk.shape[0] != gk.shape[1]:
        raise ValueError('Invalid filter! {}'.format(gk))

    kernel = gk.shape[0]
    gfilter = repeat_2dfilter(gk, 3)
    if hsv:
        gfilter[:, :, 1:2, 1:2] = 0
        gfilter[2, 2, 1:2, 1:2] = 1

    gkk = tf.constant(gfilter, tf.float32)
    pad = kernel // 2

    y = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'SYMMETRIC')

    if hsv:
        y = tf.image.rgb_to_hsv(y)

    y = tf.nn.conv2d(y, gkk, [1, 1, 1, 1], 'VALID')

    if hsv:
        y = tf.image.hsv_to_rgb(y)

    return tf.clip_by_value(y, 0, 1)


def residual(x, hsv=False):
    # Prepare the sharpening filter
    gk = np.array([[-0.0833, -0.1667, -0.0833], [-0.1667, 1, -0.1667], [-0.0833, -0.1667, -0.0833]])

    if gk is None or gk.ndim != 2 or gk.shape[0] != gk.shape[1]:
        raise ValueError('Invalid filter! {}'.format(gk))

    kernel = gk.shape[0]
    gfilter = repeat_2dfilter(gk, 3)
    if hsv:
        gfilter[:, :, 1:2, 1:2] = 0
        gfilter[2, 2, 1:2, 1:2] = 1

    gkk = tf.constant(gfilter, tf.float32)

    y = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')

    if hsv:
        y = tf.image.rgb_to_hsv(y)

    y = tf.nn.conv2d(y, gkk, [1, 1, 1, 1], 'VALID')

    if hsv:
        y = tf.image.hsv_to_rgb(y)

    return y


def _strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.compat.v1.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, 'ascii')
    return strip_def


def show_model(model, show_shapes=True, expand_nested=False):
    """ Generate a static diagram of a tf.keras.Model. """
    return tf.keras.utils.plot_model(model, show_shapes=show_shapes, expand_nested=expand_nested, dpi=72)


def show_graph(graph_def=None, width=1200, height=800, max_const_size=32, ungroup_gradients=False):
    """ Generate a dynamic visualization of a tf.keras.Model using Tensorboard. """

    if isinstance(graph_def, tf.keras.Model):
        graph_def = graph_def.inputs[0].graph.as_graph_def()

    if not graph_def:
        graph_def = tf.compat.v1.get_default_graph().as_graph_def()

    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = _strip_consts(graph_def, max_const_size=max_const_size)
    data = str(strip_def)

    if ungroup_gradients:
        data = data.replace('"gradients/', '"b_')

    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:{height}px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(data), height=height, id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:{}px;height:{}px;border:0" srcdoc="{}"></iframe>
    """.format(width, height, code.replace('"', '&quot;'))
    display(HTML(iframe))


def soft_quantization(x, alpha=255):
    """
     Quantizes a float image with values in [0,1] to simulate uint8 representation.
    """
    x = alpha * x
    x_ = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi))
    return tf.add(tf.stop_gradient(tf.round(x) - x_), x_) / alpha


def quantize_and_clip(x):
    """
    Pixel intensity rounding and clipping:
    1. Quantizes a float image with values in [0,1] to simulate uint8 representation.
    2. Clip values to [0, 1].
    :param x: image tensor
    """
    return tf.clip_by_value(soft_quantization(x), 0, 1)


def entropy(values, codebook, v=50, gamma=25):
    """
    Differentiable entropy approximation. Estimates the entropy of values quantized according to a given codebook.
    See: https://openreview.net/forum?id=HyxG3p4twS

    :param values: values to be quantized
    :param codebook: quantization code-book
    :param v: degrees of freedom for the t-Student kernel; or 0 for the Gaussian kernel
    :param gamma: gamma parameter of the kernel
    :return:
    """
    # For Gaussian, the best parameters are v=0 and gamma=5
    # for t-Student, the best parameters are v=50 and gamma=25

    # t-Student degrees of freedom
    eps = 1e-72
    prec_dtype = tf.float64

    assert (codebook.shape[0] == 1)
    assert (codebook.shape[1] > 1)

    values = tf.reshape(values, (-1, 1))

    # Compute soft-quantization
    if v <= 0:
        dff = tf.cast(values, dtype=prec_dtype) - tf.cast(codebook, dtype=prec_dtype)
        weights = tf.exp(-gamma * tf.pow(dff, 2))
    else:
        # t-Student-like distance measure with heavy tails
        dff = tf.cast(values, dtype=prec_dtype) - tf.cast(codebook, dtype=prec_dtype)
        dff = gamma * dff
        weights = tf.pow((1 + tf.pow(dff, 2) / v), -(v + 1) / 2)

    weights = (weights + eps) / (tf.reduce_sum(weights + eps, axis=1, keepdims=True))
    assert (weights.shape[1] == np.prod(codebook.shape))

    # Compute soft histogram
    histogram = tf.reduce_mean(weights, axis=0)
    histogram = tf.clip_by_value(histogram, 1e-9, tf.float32.max)
    histogram = histogram / tf.reduce_sum(histogram)
    entropy = - tf.reduce_sum(histogram * tf.math.log(histogram)) / 0.6931  # 0.6931 - log(2)
    entropy = tf.cast(entropy, tf.float32)

    return entropy, histogram, weights


def print_versions():
    print('Tensorflow:', tf.__version__)
    print('GPUs:', tf.config.list_physical_devices('GPU'))

def disable_warnings():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def disable_gpu():
    tf.config.set_visible_devices([], 'GPU')
