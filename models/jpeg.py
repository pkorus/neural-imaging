import tensorflow as tf
import numpy as np
from helpers.utils import jpeg_qtable


class DJPG:
    """
    TF model for (a differentiable) approximation of JPEG compression.
    """

    def __init__(self, sess=None, graph=None, x=None, nip_input=None, quality=None, rounding_approximation='sin',
                 rounding_approximation_steps=5):
        """
        Creates a JPEG approximation model.

        Sample usage (separately):

            jpg = DJPG()
            batch_y = jpg.process(batch_x, quality=50)
            
        Sample usage (plugged in after a NIP model):
            
            nip = UNet(sess, tf.get_default_graph(), patch_size=patch_size, loss_metric='L2')
            jpg = DJPG(sess, tf.get_default_graph(), nip.y, nip.x, quality=50, rounding_approximation='sin')
            ...
            fan = FAN(sess, tf.get_default_graph(), n_classes=2, x=imb_out, nip_input=model_a.x, n_convolutions=4)

        :param sess: TF session or None (creates a new one)
        :param graph: TF graph or None (creates a new one)
        :param x: input to the DJPG module (TF tensor) or None (creates a placeholder)
        :param nip_input: input to the NIP (if part of a larger model) or None
        :param quality: JPEG quality level or None (can be specified later)
        :param rounding_approximation: None (uses normal rounding), 'sin', 'soft', or 'harmonic'
        :param rounding_approximation_steps: number of approximation terms (for 'harmonic' approx. only)
        """

        # Sanitize inputs
        if rounding_approximation is not None and rounding_approximation not in ['sin', 'harmonic', 'soft']:
            raise ValueError('Unsupported rounding approximation: {}'.format(rounding_approximation))

        # Remember settings
        self.rounding_approximation = rounding_approximation
        self.rounding_approximation_steps = rounding_approximation_steps
        self.init_quality = quality
        block_size = 8

        # Configure TF objects
        self.graph = graph or tf.Graph()
        self.sess = sess or tf.Session(graph=self.graph)

        with self.graph.as_default():
            with tf.name_scope('jpeg'):
                if x is None:
                    x = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='jpeg_x')
                    self.use_nip_input = False
                else:
                    self.use_nip_input = True
    
                # Color conversion (RGB -> YCbCr)
                with tf.name_scope('rgb_to_ycbcr'):
                    # RGB to YCbCr conversion
                    color_F = tf.constant([[0, 0.299, 0.587, 0.114], [128, -0.168736, -0.331264, 0.5], [128, 0.5, -0.418688, -0.081312]])
                    color_I = tf.constant([[-1.402 * 128, 1, 0, 1.402], [1.058272 * 128, 1, -0.344136, -0.714136], [-1.772 * 128, 1, 1.772, 0]])
                                
                    xc = tf.pad(255.0 * x, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=1)
                    ycbcr = tf.nn.conv2d(xc, tf.reshape(tf.transpose(color_F), [1, 1, 4, 3]), [1, 1, 1, 1], 'SAME', name='jpeg_ycbcr')
    
                with tf.name_scope('blocking'):
                    # Re-organize to get non-overlapping blocks in the following form
                    # (n_examples * 3, block_size, block_size, n_blocks)
                    p = tf.transpose(ycbcr - 127, [0, 3, 1, 2])
                    p = tf.reshape(p, [-1, tf.shape(p)[2], tf.shape(p)[3]])
                    p = tf.expand_dims(p, axis=3)
                    p = tf.space_to_depth(p, block_size)
                    p = tf.transpose(p, [0, 3, 1, 2])
                    p = tf.reshape(p, [-1, block_size, block_size, tf.shape(p)[2] * tf.shape(p)[3]])

                    # Reorganize to move n_blocks to the first dimension
                    r = tf.transpose(p, [0, 3, 1, 2])
                    r = tf.reshape(r, [-1, r.shape[2], r.shape[3]])
    
                # Forward DCT transform
                with tf.name_scope('dct'):
                    # DCT transformation matrix
                    dct_F = tf.constant([[0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536, 0.3536],
                                         [0.4904, 0.4157, 0.2778, 0.0975, -0.0975, -0.2778, -0.4157, -0.4904],
                                         [0.4619, 0.1913, -0.1913, -0.4619, -0.4619, -0.1913, 0.1913, 0.4619],
                                         [0.4157, -0.0975, -0.4904, -0.2778, 0.2778, 0.4904, 0.0975, -0.4157],
                                         [0.3536, -0.3536, -0.3536, 0.3536, 0.3536, -0.3536, -0.3536, 0.3536],
                                         [0.2778, -0.4904, 0.0975, 0.4157, -0.4157, -0.0975, 0.4904, -0.2778],
                                         [0.1913, -0.4619, 0.4619, -0.1913, -0.1913, 0.4619, -0.4619, 0.1913],
                                         [0.0975, -0.2778, 0.4157, -0.4904, 0.4904, -0.4157, 0.2778, -0.0975]])
                    dct_I = tf.transpose(dct_F)                    
                
                    Xi = tf.matmul(tf.tile(tf.expand_dims(dct_F, axis=0), [tf.shape(r)[0], 1, 1]), r)
                    X = tf.matmul(Xi, tf.tile(tf.expand_dims(dct_I, axis=0), [tf.shape(r)[0], 1, 1]), name='jpeg_dct')
    
                # Approximate quantization
                with tf.name_scope('quantization'):
                    # JPEG Quantization tables
                    if quality is None:
                        Q_mtx_lum = tf.placeholder(tf.float32, (8, 8))
                        Q_mtx_chr = tf.placeholder(tf.float32, (8, 8))
                    else:
                        Q_mtx_lum = tf.constant(jpeg_qtable(quality, 0))
                        Q_mtx_chr = tf.constant(jpeg_qtable(quality, 1))
    
                    # Tile quantization values for successive channels: 
                    # image_0 [R .. R G .. G B .. B] ... image_N [R .. R G .. G B .. B]
                    Ql = tf.tile(tf.expand_dims(Q_mtx_lum, axis=0), [1 * (tf.shape(p)[-1]), 1, 1])
                    Qc = tf.tile(tf.expand_dims(Q_mtx_chr, axis=0), [2 * (tf.shape(p)[-1]), 1, 1])
                    Q = tf.concat((Ql, Qc), axis=0)
                    Q = tf.tile(Q, [(tf.shape(x)[0]), 1, 1])
                    X = X / Q

                    if rounding_approximation is None:
                        X = tf.round(X)
                    elif rounding_approximation == 'sin':
                        X = X - tf.sin(2 * np.pi * X) / (2 * np.pi)
                    elif rounding_approximation == 'soft':
                        XA = X - tf.sin(2 * np.pi * X) / (2 * np.pi)
                        X = tf.stop_gradient(tf.round(X) - XA) + XA
                    elif rounding_approximation == 'harmonic':
                        XA = X - tf.sin(2 * np.pi * X) / np.pi
                        for k in range(2, rounding_approximation_steps):
                            XA += tf.pow(-1.0, k) * tf.sin(2 * np.pi * k * X) / (k * np.pi)
                        X = XA

                    X = X * Q
    
                with tf.name_scope('idct'):
                    # Inverse DCT transform
                    xi = tf.matmul(tf.tile(tf.expand_dims(dct_I, axis=0), [tf.shape(r)[0], 1, 1]), X)
                    xi = tf.matmul(xi, tf.tile(tf.expand_dims(dct_F, axis=0), [tf.shape(r)[0], 1, 1]))
    
                with tf.name_scope('rev-blocking'):
                    # Reorganize data back to
                    xi = tf.reshape(xi, [3 * tf.shape(x)[0], -1, xi.shape[1], xi.shape[2]])
                    xi = tf.transpose(xi, [0, 2, 3, 1])

                    # Backward re-organization from blocks
                    # (n_examples * 3, block, block, n_blocks) -> (n_examples, w, h, 3)
                    q = tf.reshape(xi, [-1, tf.shape(xi)[1] * tf.shape(xi)[2], tf.shape(x)[1] // block_size,
                                        tf.shape(x)[2] // block_size])
                    q = tf.transpose(q, [0, 2, 3, 1])
                    q = tf.depth_to_space(q, block_size)
                    q = tf.reshape(q, [-1, 3, tf.shape(q)[1], tf.shape(q)[2]])
                    q = tf.transpose(q, [0, 2, 3, 1])
    
                # Color conversion (YCbCr-> RGB)
                with tf.name_scope('ycbcr_to_rgb'):
                    qc = tf.pad(q + 127, [[0, 0], [0, 0], [0, 0], [1, 0]], 'CONSTANT', constant_values=1)
                    y = tf.nn.conv2d(qc, tf.reshape(tf.transpose(color_I), [1, 1, 4, 3]), [1, 1, 1, 1], 'SAME', name='jpeg_y')                                                
                    y = y / 255.0                    
                    y = tf.clip_by_value(y, 0, 1)

        self.x = x
        self.y = y
        self.nip_input = nip_input
        self.Q_mtx_lum = Q_mtx_lum
        self.Q_mtx_chr = Q_mtx_chr

    def process(self, batch_x, quality=None):
        with self.graph.as_default():
            if quality is None:
                y = self.sess.run(self.y, feed_dict={
                    self.x if not self.use_nip_input else self.nip_input: batch_x,
                })
            else:
                y = self.sess.run(self.y, feed_dict={
                    self.x if not self.use_nip_input else self.nip_input: batch_x,
                    self.Q_mtx_lum: jpeg_qtable(quality, 0),
                    self.Q_mtx_chr: jpeg_qtable(quality, 1)
                })
            return y

    def __repr__(self):
        if self.rounding_approximation == 'harmonic':
            return 'DJPG(rounding_approximation={}, rounding_approximation_steps={})'.format(self.rounding_approximation, self.rounding_approximation_steps)
        else:
            return 'DJPG(rounding_approximation={})'.format(self.rounding_approximation)

