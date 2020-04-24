# -*- coding: utf-8 -*-
"""
Various kernels for image filtering.
"""
import numpy as np
from scipy import signal


def upsampling_kernel(cfa_pattern='gbrg'):
    """
    Ideal initialization of up-sampling kernels for matching the 12-feature-layer format needed by depth-to-space.
    :param cfa_pattern: CFA pattern, e.g., 'GBRG'
    """
    cfa_pattern = cfa_pattern.upper()

    if cfa_pattern.upper() == 'GBRG':
        #                R  G  B  R  G  B  R  G  B  R  G  B
        #                1  1  1  2  2  2  3  3  3  4  4  4
        upk = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                       ])
    elif cfa_pattern.upper() == 'RGGB':
        #                R  G  B  R  G  B  R  G  B  R  G  B
        #                1  1  1  2  2  2  3  3  3  4  4  4
        upk = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                       ])
    elif cfa_pattern.upper() == 'BGGR':
        #                R  G  B  R  G  B  R  G  B  R  G  B
        #                1  1  1  2  2  2  3  3  3  4  4  4
        upk = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                       ])
    else:
        raise ValueError('Unsupported CFA pattern: {}'.format(cfa_pattern))

    return upk


def gamma_kernels():
    """
    Pre-trained kernels of a toy neural network for approximation of gamma correction.
    """
    gamma_dense1_kernel = np.array([2.9542332, 17.780445, 0.6280197, 0.40384966])
    gamma_dense1_bias = np.array([0.4047071, 1.1489044, -0.17624384, 0.47826886])

    gamma_dense2_kernel = np.array([0.44949612, 0.78081024, 0.97692937, -0.24265033])
    gamma_dense2_bias = np.array([-0.4702738])

    gamma_d1k = np.zeros((3, 12))
    gamma_d1b = np.zeros((12, ))
    gamma_d2k = np.zeros((12, 3))
    gamma_d2b = np.zeros((3,))

    for r in range(3):
        gamma_d1k[r, r*4:r*4+4] = gamma_dense1_kernel
        gamma_d1b[r*4:r*4+4] = gamma_dense1_bias
        gamma_d2k[r*4:r*4+4, r] = gamma_dense2_kernel
        gamma_d2b[r] = gamma_dense2_bias

    return gamma_d1k, gamma_d1b, gamma_d2k, gamma_d2b


def bilin_kernel(kernel=3):
    """
    Bilinear demosaicing kernel.
    """
    g_kern = np.array([[0, 1/4., 0], [1/4., 1, 1/4.], [0, 1/4., 0]])
    rb_kern = np.array([[1/4., 1/2., 1/4.], [1/2., 1, 1/2.], [1/4., 1/2., 1/4.]])

    G_kern = np.zeros((3,3,3), np.float32)
    G_kern[:, :, 1] = g_kern

    R_kern = np.zeros((3,3,3), np.float32)
    R_kern[:, :, 0] = rb_kern

    B_kern = np.zeros((3,3,3), np.float32)
    B_kern[:, :, 2] = rb_kern

    dmf = np.stack((R_kern, G_kern, B_kern), axis=3)
    if kernel > 3:
        pad = (kernel - 3) // 2
        dmf = np.pad(dmf, ((pad, pad), (pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)

    return dmf


def gkern(kernlen=5, std=0.83):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


def repeat_2dfilter(f, channels=3, pad=0):
    """
    Repeat a 2D filter along channel dimensions (both input/output). Useful for kernel initialization in conv. layers.
    :param f: 2d filter
    :param channels: number of input/output channels
    :param pad: optional padding (along the spatial dimension)
    :return: valid conv 2d kernel (kernel, kernel, channels, channels)
    """
    rf = np.zeros((f.shape[0] + 2 * pad, f.shape[1] + 2 * pad, channels, channels))

    for r in range(channels):
        rf[:, :, r, r] = np.pad(f, [pad, pad], 'constant')

    return rf


def center_mask_2dfilter(f_size, channels):
    indicator = np.zeros((f_size, f_size, channels, channels))

    for r in range(channels):
        indicator[f_size // 2, f_size // 2, r, r] = 1

    return indicator
