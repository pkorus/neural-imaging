import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim as slim

from helpers.utils import gkern, repeat_2dfilter
from IPython.display import display, HTML

activation_mapping = {
    'leaky_relu' : tf.nn.leaky_relu,
    'relu': tf.nn.relu,
    'tanh': tf.nn.tanh,
    'sigmoid': tf.nn.sigmoid,
    'softsign': tf.nn.softsign
}


def manipulation_resample(x, factor=2):
    with tf.name_scope('resample'):
        im_res = tf.image.resize_images(x, [tf.shape(x)[1] // factor, tf.shape(x)[1] // factor])
        return tf.image.resize_images(im_res, [tf.shape(x)[1], tf.shape(x)[1]])


def manipulation_awgn(x, strength=0.025):
    with tf.name_scope('awgn'):
        im_awgn = x + strength * tf.random.normal(tf.shape(x))
        im_awgn = quantization(255.0 * im_awgn, 'quantization', 'quantized', 'soft')
        return im_awgn / 255.0


def manipulation_gamma(x, strength=2.0):
    with tf.name_scope('gamma'):
        im_gamma = tf.pow(x, strength, name='squared')
        im_gamma = quantization(255.0 * im_gamma, 'quantization', 'quantized', 'soft')
        return tf.pow(tf.clip_by_value(im_gamma, 1, 255) / 255.0, 1/strength, name='sqrt')


def manipulation_median(x, kernel):
    with tf.name_scope('median_filter'):
        xp = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')
        patches = tf.extract_image_patches(xp, [1, kernel, kernel, 1], [1, 1, 1, 1], 4*[1], 'VALID')
        patches = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2], tf.shape(patches)[3]//3, 3])
        return tf.contrib.distributions.percentile(patches, 50, axis=3)

    
def manipulation_gaussian(x, kernel, std, skip_clip=False):
    with tf.name_scope('gaussian_filter'):
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


def manipulation_sharpen(x, filter=None, hsv=True):
    with tf.name_scope('sharpen_filter'):

        if filter == 0:  # weak
            gk = np.array([[-0.0833, -0.1667, -0.0833], [-0.1667, 2.0000, -0.1667], [-0.0833, -0.1667, -0.0833]])
        elif filter == 1:  # strong
            gk = np.array([[-0.1667, -0.6667, -0.1667], [-0.6667, 4.3333, -0.6667], [-0.1667, -0.6667, -0.1667]])
        else:
            gk = None

        # Other standard filters
        # gk = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # gk = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])            

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

        return tf.clip_by_value(y, 0, 1)


def memory_usage_tf(sess):
    return sess.run(tf.contrib.memory_stats.BytesInUse())            


def memory_usage_tf_variables(global_vars=True):
    bytes = 0
    for tv in (tf.trainable_variables() if not global_vars else tf.global_variables()):
        bytes += np.prod(tv.shape.as_list()) * tv.dtype.size
    return bytes


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = bytes("<stripped %d bytes>"%size, 'ascii')
    return strip_def


def show_graph(graph_def=None, width=1200, height=800, max_const_size=32, ungroup_gradients=False):
    if not graph_def:
        graph_def = tf.get_default_graph().as_graph_def()
        
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
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
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(data), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:{}px;height:{}px;border:0" srcdoc="{}"></iframe>
    """.format(width, height, code.replace('"', '&quot;'))
    display(HTML(iframe))


def quantization(x, scope, name, rounding='soft', approx_steps=1, codebook_tensor=None, v=50, gamma=25):
    
    with tf.name_scope(scope):
        
        if rounding is None:
            x = tf.round(x)
        
        elif rounding == 'sin':
            x = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi), name=name)
        
        elif rounding == 'soft':
            x_ = tf.subtract(x, tf.sin(2 * np.pi * x) / (2 * np.pi), name='{}_soft'.format(name))
            x = tf.add(tf.stop_gradient(tf.round(x) - x_), x_, name=name)
        
        elif rounding == 'harmonic':
            xa = x - tf.sin(2 * np.pi * x) / np.pi
            for k in range(2, approx_steps):
                xa += tf.pow(-1.0, k) * tf.sin(2 * np.pi * k * x) / (k * np.pi)
            x = tf.identity(xa, name=name)
            
        elif rounding == 'identity':
            x = x
            
        elif rounding == 'soft-codebook':
            
            prec_dtype = tf.float64
            eps = 1e-72
            
            assert(codebook_tensor.shape[0] == 1)
            assert(codebook_tensor.shape[1] > 1)            

            values = tf.reshape(x, (-1, 1))

            if v <= 0:
                # Gaussian soft quantization
                weights = tf.exp(-gamma * tf.pow(tf.cast(values, dtype=prec_dtype) - tf.cast(codebook_tensor, dtype=prec_dtype), 2))
            else:
                # t-Student soft quantization
                dff = tf.cast(values, dtype=prec_dtype) - tf.cast(codebook_tensor, dtype=prec_dtype)
                dff = gamma * dff
                weights = tf.pow((1 + tf.pow(dff, 2)/v), -(v+1)/2)
            
            weights = (weights + eps) / (tf.reduce_sum(weights + eps, axis=1, keepdims=True))
            
            assert(weights.shape[1] == np.prod(codebook_tensor.shape))

            soft = tf.reduce_mean(tf.matmul(weights, tf.transpose(tf.cast(codebook_tensor, dtype=prec_dtype))), axis=1)
            soft = tf.cast(soft, dtype=tf.float32)
            soft = tf.reshape(soft, tf.shape(x))            

            hard = tf.gather(codebook_tensor, tf.argmax(weights, axis=1), axis=1)
            hard = tf.reshape(hard, tf.shape(x))            

            x = tf.stop_gradient(hard - soft) + soft
            x = tf.identity(x, name=name)
        
        else:
            raise ValueError('Unknown quantization! {}'.format(rounding))
    
    return x


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels, name='upsampling_kernel', scope=None):
    with tf.name_scope(scope):
        pool_size = 2
        deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02), name=name)
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])
        deconv_output = tf.concat([deconv, x2], 3)
        deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


def nm(x):
    w0 = tf.Variable(1.0, name='w0')
    w1 = tf.Variable(0.0, name='w1')
    return w0*x + w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it


def entropy(values, codebook, v=50, gamma=25):

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
    entropy = - tf.reduce_sum(histogram * tf.log(histogram)) / 0.6931  # 0.6931 - log(2)
    entropy = tf.cast(entropy, tf.float32)

    return entropy, histogram, weights
