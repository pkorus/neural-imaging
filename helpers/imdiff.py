# -*- coding: utf-8 -*-
"""
Utility functions for comparing images (return matplotlib figures).
"""
import numpy as np

from helpers import plots, metrics, image
from helpers.image import fft_log_norm


def compare_images_ab_ref(img_ref, img_a, img_b, labels=None, extras=False, fig=None):

    labels = labels or ['target', '', '']

    img_a = img_a.squeeze()
    img_b = img_b.squeeze()
    img_ref = img_ref.squeeze()

    fig, axes = plots.sub(9 if extras else 3, ncols=3, fig=fig)
    # Index of the last axes 
    j = 3 if extras else 2

    plots.image(img_ref, '(T) {}'.format(labels[0]), axes=axes[0])

    label_a = '(A) {}: {:.1f} dB / {:.3f}'.format(labels[1], metrics.psnr(img_ref, img_a), metrics.ssim(img_ref, img_a))
    plots.image(img_a, label_a, axes=axes[1])

    label_b = '(B) {}: {:.1f} dB / {:.3f}'.format(labels[2], metrics.psnr(img_ref, img_b), metrics.ssim(img_ref, img_b))
    plots.image(img_b, label_b, axes=axes[j])

    # A hack to allow image axes to zoom together
    axes[1].get_shared_x_axes().join(axes[0], axes[1])
    axes[j].get_shared_x_axes().join(axes[0], axes[j])
    axes[1].get_shared_y_axes().join(axes[0], axes[1])
    axes[j].get_shared_y_axes().join(axes[0], axes[j])

    if not extras:
        return fig

    # Compute and plot difference images
    diff_a = np.abs(img_a - img_ref)
    diff_a_mean = diff_a.mean()
    diff_a = image.normalize(diff_a, 0.1)

    diff_b = np.abs(img_b - img_ref)
    diff_b_mean = diff_b.mean()
    diff_b = image.normalize(diff_b, 0.1)

    diff_ab = np.abs(img_b - img_a)
    diff_ab_mean = diff_ab.mean()
    diff_ab = image.normalize(diff_ab, 0.1)

    plots.image(diff_a, 'T - A: mean abs {:.3f}'.format(diff_a_mean), axes=axes[2])
    plots.image(diff_b, 'T - B: mean abs {:.3f}'.format(diff_b_mean), axes=axes[6])
    plots.image(diff_ab, 'A - B: mean abs {:.3f}'.format(diff_ab_mean), axes=axes[4])

    # Compute and plot spectra
    fft_a = fft_log_norm(diff_a)
    fft_b = fft_log_norm(diff_b)

    # fft_ab = utils.normalize(np.abs(fft_a - fft_b))
    fft_ab = image.normalize(np.abs(fft_log_norm(img_b) - fft_log_norm(img_a)), 0.01)
    plots.image(fft_a, 'FFT(T - A)', axes=axes[5])
    plots.image(fft_b, 'FFT(T - B)', axes=axes[7])
    plots.image(fft_ab, 'FFT(A) - FFT(B)', axes=axes[8])

    return fig


def compare_batches(batch_a, batch_b, labels=None, fig=None, figwidth=4, nrows=3, transpose=False):

    n_images = min(len(batch_a), len(batch_b))

    labels = labels or ['', '']

    fig, axes = plots.sub(n_images * nrows, figwidth, ncols=n_images, fig=fig, transpose=transpose)

    for i, (img_a, img_b) in enumerate(zip(batch_a, batch_b)):

        label_a = f'(A) {labels[0]}'
        plots.image(img_a, label_a, axes=axes[i + n_images * 0])

        psnr = metrics.psnr(img_a, img_b)
        ssim = metrics.ssim(img_a, img_b)
        label_b = f'(B) {labels[1]}: {psnr:.1f} dB / {ssim:.3f}'
        plots.image(img_b, label_b, axes=axes[i + n_images * 1])

        diff_ab = img_b - img_a
        diff_abs = np.abs(img_b - img_a)
        diff_mean = diff_ab.mean()

        if nrows > 2:
            plots.image(image.normalize(diff_ab), f'A - B: mean abs {diff_mean:.3f}', axes=axes[i + n_images*2])
        
        if nrows > 3:
            plots.image(image.normalize(diff_abs, 0.1), f'|A - B|: mean abs {diff_mean:.3f}', axes=axes[i + n_images*3])

    return fig
