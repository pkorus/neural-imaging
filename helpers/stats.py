# -*- coding: utf-8 -*-
"""
Common performance metrics & statistics (accuracy, tpr, auc) and helper functions (moving average).
"""
import numpy as np
from scipy import stats


def detection_accuracy(positive, negative, bins=100, return_index=False):
    """
    Estimate binary detection accuracy from response distributions for matching and missing samples. Uses a simple
    threshold for decision and finds the best accuracy:

        0.5 * (np.mean(H_match >= t) + np.mean(H_miss < t))

    :param positive: positive detection scores (numpy array)
    :param negative: negative detection scores (numpy array)
    :param bins: number / list of candidate thresholds to consider (int or iterable)
    :param return_index: flag to return the index of the threshold instead of the threshold itself

    :returns tuple (best accuracy, threshold / index)
    """

    if isinstance(bins, int):
        bins = span(negative, positive, bins)

    accuracies = [0.5 * (np.mean(positive >= thresh) + np.mean(negative < thresh)) for thresh in bins]

    if return_index:
        return max(accuracies), np.argmax(accuracies)
    else:
        return max(accuracies), bins[np.argmax(accuracies)]


def true_positive_rate(positive, negative, fpr=0.01):
    """
    Estimate true positive rate at a fixed false positive rate threshold.
    :param positive: positive detection scores (numpy array)
    :param negative: negative detection scores (numpy array)
    :param fpr: false positive rate (0-1)
    :return:
    """
    thresh = np.percentile(negative, 100 * (1 - fpr))
    return np.mean(positive >= thresh)


def roc(positive, negative, bins=100):
    """
    Returns tpr, fpr coordinates along an ROC curve.
    :param positive: positive detection scores (numpy array)
    :param negative: negative detection scores (numpy array)
    :param bins: number of candidate thresholds to consider (int)
    :return: tuple of arrays (tpr, fpr)
    """
    cc = span(negative, positive, bins)
    tpr = [np.mean(positive >= t) for t in cc][::-1]
    fpr = [np.mean(negative >= t) for t in cc][::-1]
    return tpr, fpr


def auc(positive, negative, bins=100):
    """
    Returns an approximate AUC (trapezoid ROC integration).

    :param positive: positive detection scores (numpy array)
    :param negative: negative detection scores (numpy array)
    :param bins: number of candidate thresholds to consider (int)
    :return: auc
    """
    tpr, fpr = roc(positive, negative, bins)

    if tpr[0] != 0 or fpr[0] != 0:
        raise ValueError('The ROC should start at (0, 0) - double check the detection threshold sweep')

    if tpr[-1] != 1 or fpr[-1] != 1:
        raise ValueError('The ROC should end at (1, 1) - double check the detection threshold sweep')

    return np.trapz(tpr, fpr)


def inlier_rate(candidates, reference, perc=0.05):
    """
    Counts the ratio of candidate points that fall within the bottom and top percentiles of a reference distribution.
    :param candidates: samples from the candidate distribution (numpy array)
    :param reference: samples from the reference distribution (numpy array)
    :param perc: percentile (0-1)
    :return: fraction of candidate points (float in 0-1)
    """
    return np.mean((candidates > np.percentile(reference, 100 * perc)) * (candidates < np.percentile(reference, 100 * (1 - perc))))


def corrcoeff(a, b):
    """ Returns the normalized correlation coefficient between two arrays """
    a = (a - np.mean(a)) / (1e-9 + np.std(a))
    b = (b - np.mean(b)) / (1e-9 + np.std(b))
    return np.mean(a * b)


def rsquared(a, b):
    """ Returns the coefficient of determination (R^2) between two arrays (normalized) """
    from sklearn.metrics import r2_score
    a = (a - np.mean(a)) / (1e-9 + np.std(a))
    b = (b - np.mean(b)) / (1e-9 + np.std(b))
    return r2_score(a, b)


def hist(values, code_book, density=False):
    """
    Returns a histogram of values quantized to given centroids (as opposed to numpy which uses bin edges).
    :param values: values to be quantized
    :param code_book: quantization code-book (centroids)
    :param density:
    :return:
    """
    f = np.histogram(values.ravel(), bins=bin_edges(code_book), density=density)[0]
    return f if not density else f / f.sum()


def entropy(samples, code_book=None):
    """
    Estimate entropy of the samples quantized to given centroids.
    :param samples: data samples
    :param code_book: quantization code-book (centroids)
    :return: entropy
    """
    if code_book is None:
        code_book = np.arange(-255, 255, 1).reshape((-1,))
    counts = hist(samples, code_book)
    counts = counts.clip(min=1)
    probs = counts / counts.sum()
    return - np.sum(probs * np.log2(probs))


def bin_edges(code_book):
    max_float = np.abs(code_book).max() * 2
    code_book_edges = np.convolve(code_book, [0.5, 0.5], mode='valid')
    code_book_edges = np.concatenate((-np.array([max_float]), code_book_edges, np.array([max_float])), axis=0)
    return code_book_edges


def kld_discrete(samples_a, samples_b, bins=25):
    cc = span(samples_a, samples_b, bins)

    p1 = hist(samples_a, cc, density=True).clip(min=1e-16)
    p2 = hist(samples_b, cc, density=True).clip(min=1e-16)

    return stats.entropy(p1, p2)


def span(negative, positive, bins=100):
    bins = np.linspace(np.min([positive.min(), negative.min()]) - 1e-6,
                       np.max([positive.max(), negative.max()]) + 1e-6, bins)
    return bins


def ma_gaussian(x, y, step_size=0.05, width=10):
    """Moving average with Gaussian averaging"""
    bin_centers = np.arange(np.min(x), np.max(x) - 0.5*step_size, step_size) + 0.5*step_size
    bin_avg = np.zeros(len(bin_centers))

    # We're going to weight with a Gaussian function
    def gaussian(x, amp=1, mean=0, sigma=1):
        return amp*np.exp(-(x-mean)**2/(2*sigma**2))

    for index in range(0, len(bin_centers)):
        bin_center = bin_centers[index]
        weights = gaussian(x, mean=bin_center, sigma=width)
        bin_avg[index] = np.average(y, weights=weights)

    return bin_centers, bin_avg


def ma_conv(x, n=10):
    """Moving average with a simple box filter."""

    if len(x) == 0:
        return np.array([])

    if n == 0:
        n = (len(x) // 10)

    fn = 2*n + 1

    return np.convolve(np.pad(x, n, 'edge'), np.ones((fn,))/fn, mode='valid')


def ma_exp(x, alpha=0.1):
    """ Exponential moving average """
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, x.shape[0]):
        y[i] = alpha * x[i] + (1-alpha) * y[i-1]

    return y
