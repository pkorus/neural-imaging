#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import scipy as sp
import numpy as np

def get_sample_images(dataset):

    if 'clic' in dataset:
        return [0, 13, 33, 36]
    
    if 'kodak' in dataset:
        return [4, 14, 20, 22]
    
    if 'raw' in dataset:
        return [11, 19, 34, 35]

def spectrum(x, domain='fft', mf=0, gray=False):
    import scipy.fftpack as sfft

    x = x.squeeze()
    if x.ndim != 3:
        raise ValueError('Only single images can be accepted as input.')
    y = np.zeros_like(x)
    for i in range(x.shape[-1]):

        if domain == 'dct':
            y[:, :, i] = np.abs(sp.fftpack.dct(sp.fftpack.dct(x[:, :, i].T).T))
        elif domain == 'fft':
            y[:, :, i] = np.abs(sfft.fft2(x[:, :, i]))
            y[:, :, i] = sfft.fftshift(y[:, :, i])
        else:
            raise ValueError('Invalid domain!')

        if mf > 1:
            y[:, :, i] = sp.ndimage.median_filter(y[:, :, i], 15)

    return y if not gray else y.mean(axis=2)