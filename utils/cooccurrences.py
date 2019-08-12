import scipy as sp
import itertools
import numpy as np

filters = {
  'h' : np.array([[-1, 1]]),
  'v' : np.array([[-1], [1]]),
  'd1' : np.array([[-1, 0], [0, 1]]),
  'd2' : np.array([[0, -1], [1, 0]]),
  'f': np.array([[1, -3], [3, -1]]),
  'r': np.array([[-0.0833, -0.1667, -0.0833], [-0.1667, 1.0000, -0.1667], [-0.0833, -0.1667, -0.0833]])
}


def get_block_features(image, t=1, q=2, order=4, f='f'):

    residual = sp.signal.convolve2d(image, filters[f], 'same')
    residual = np.round(residual / q)

    # Truncation
    residual[residual > t] = t
    residual[residual < -t] = -t

    # Generate shifted versions of the residual for easier computation of co-occurrences
    shifted = []
    for r in range(order):
        shifted.append(residual[:, r:(-order+r)])

    # Helper / storage variables
    cooc = np.zeros(np.repeat(2*t + 1, order))
    comb_gen = itertools.product(range(-t, t+1), repeat=order)

    # Generate co-occurrence statistics
    for comb in comb_gen:
        c = np.array(comb)
        index = c + t
        pattern = shifted[0] == c[0]
        for r in range(1, order):
            pattern &= shifted[r] == c[r]
        cooc[tuple(index)] = np.mean(pattern)

    # Return a feature vector
    vec = cooc.reshape((np.product(cooc.shape), ))
    return vec
