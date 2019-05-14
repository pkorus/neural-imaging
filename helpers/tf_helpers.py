import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim as slim

from helpers.utils import gkern, repeat_2dfilter
from IPython.display import display, HTML


def tf_median(x, kernel):
    with tf.name_scope('median_filter'):
        xp = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')
        patches = tf.extract_image_patches(xp, [1, kernel, kernel, 1], [1, 1, 1, 1], 4*[1], 'VALID')
        patches = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1], tf.shape(patches)[2], tf.shape(patches)[3]//3, 3])
        return tf.contrib.distributions.percentile(patches, 50, axis=3)

    
def tf_gaussian(x, kernel, sigma, skip_clip=False):
    with tf.name_scope('gaussian_filter'):
        gk = gkern(kernel, sigma)
        gfilter = np.zeros((kernel, kernel, 3, 3))
        for r in range(3):
            gfilter[:,:,r,r] = gk
        gkk = tf.constant(gfilter, tf.float32)
        xp = tf.pad(x, [[0, 0], 2*[kernel//2], 2*[kernel//2], [0, 0]], 'REFLECT')
        y = tf.nn.conv2d(xp, gkk, [1, 1, 1, 1], 'VALID')
        if skip_clip:
            return y
        else:
            return tf.clip_by_value(y, 0, 1)


def tf_sharpen(x, filter=None, hsv=True):
    with tf.name_scope('sharpen_filter'):
        if filter == 2:
            gk = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        elif filter == 1:
            gk = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        elif filter == 0:
            gk = np.array([[-0.1667, -0.6667, -0.1667], [-0.6667, 4.3333, -0.6667], [-0.1667, -0.6667, -0.1667]])
        else:
            gk = None

        if gk is None or gk.ndim != 2 or gk.shape[0] != gk.shape[1]:
            raise ValueError('Invalid filter!')

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


def quantization(x, scope, name, rounding='soft', approx_steps=5, codebook_tensor=None, soft_quantization_sigma=1):
    
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
            x = xa
            
        elif rounding == 'identity':
            x = x
            
        elif rounding == 'soft-codebook':
            
            prec_dtype = tf.float64
            v = 100
            eps = 1e-72
            
            assert(codebook_tensor.shape[0] == 1)
            assert(codebook_tensor.shape[1] > 1)            

            values = tf.reshape(x, (-1, 1))

            if v <= 0:
                # Gaussian soft quantization
                weights = tf.exp(-soft_quantization_sigma * tf.pow(tf.cast(values, dtype=prec_dtype) - tf.cast(codebook_tensor, dtype=prec_dtype), 2)) 
            else:
                # t-Student soft quantization
                dff = tf.cast(values, dtype=prec_dtype) - tf.cast(codebook_tensor, dtype=prec_dtype)
                dff = soft_quantization_sigma * dff
                weights = tf.pow((1 + tf.pow(dff, 2)/v), -(v+1)/2)
            
            weights = (weights + eps) / (eps + tf.reduce_sum(weights, axis=1, keepdims=True))
            
            assert(weights.shape[1] == np.prod(codebook_tensor.shape))

            soft = tf.reduce_mean(tf.matmul(weights, tf.transpose(tf.cast(codebook_tensor, dtype=prec_dtype))), axis=1)
            soft = tf.cast(soft, dtype=tf.float32)
            soft = tf.reshape(soft, tf.shape(x))            

            hard = tf.gather(codebook_tensor, tf.argmax(weights, axis=1), axis=1)
            hard = tf.reshape(hard, tf.shape(x))            

            x = tf.stop_gradient(hard - soft) +  soft
        
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