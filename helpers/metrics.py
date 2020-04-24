# -*- coding: utf-8 -*-
"""
Image quality metrics.
"""
import numpy as np
from skimage import metrics


def ssim(a, b):
    if a.ndim == 4 and a.shape[0] == 1:
        a = a.squeeze()

    if b.ndim == 4 and b.shape[0] == 1:
        b = b.squeeze()

    if a.ndim == 3 and b.ndim == 3:
        return metrics.structural_similarity(a, b, multichannel=True, data_range=1)

    elif a.ndim == 4 and b.ndim == 4:
        out = np.zeros((a.shape[0],))
        for i in range(a.shape[0]):
            out[i] = ssim(a[i], b[i])
        return out

    else:
        raise ValueError('Incompatible tensor shapes! {} and {}'.format(a.shape, b.shape))


def psnr(a, b):
    if a.ndim == 4 and a.shape[0] == 1:
        a = a.squeeze()

    if b.ndim == 4 and b.shape[0] == 1:
        b = b.squeeze()

    if a.ndim == 3 and b.ndim == 3:
        return metrics.peak_signal_noise_ratio(a, b, data_range=1)

    elif a.ndim == 4 and b.ndim == 4:
        out = np.zeros((a.shape[0],))
        for i in range(a.shape[0]):
            out[i] = psnr(a[i], b[i])
        return out

    else:
        raise ValueError('Incompatible tensor shapes! {} and {}'.format(a.shape, b.shape))


def mse(a, b):
    if a.ndim == 4 and a.shape[0] == 1:
        a = a.squeeze()

    if b.ndim == 4 and b.shape[0] == 1:
        b = b.squeeze()

    if a.ndim == 3 and b.ndim == 3:
        return metrics.mean_squared_error(a, b)

    elif a.ndim == 4 and b.ndim == 4:
        out = np.zeros((a.shape[0],))
        for i in range(a.shape[0]):
            out[i] = mse(a[i], b[i])
        return out

    else:
        raise ValueError('Incompatible tensor shapes! {} and {}'.format(a.shape, b.shape))

    return metrics.mean_squared_error(a.squeeze(), b.squeeze())


def mae(a, b):
    if a.ndim == 4 and a.shape[0] == 1:
        a = a.squeeze()

    if b.ndim == 4 and b.shape[0] == 1:
        b = b.squeeze()

    if a.ndim == 3 and b.ndim == 3:
        return np.mean(np.abs(a - b))

    elif a.ndim == 4 and b.ndim == 4:
        out = np.zeros((a.shape[0],))
        for i in range(a.shape[0]):
            out[i] = mae(a[i], b[i])
        return out

    else:
        raise ValueError('Incompatible tensor shapes! {} and {}'.format(a.shape, b.shape))


def batch(a, b, metric=ssim):
    assert a.ndim == 4 and b.ndim == 4, 'Input arrays need to be 4-dim: batch, height, width, channels'
    assert len(a) == len(b), 'Image batches must be of the same length'
    return np.mean([metric(a[r], b[r]) for r in range(len(a))])
