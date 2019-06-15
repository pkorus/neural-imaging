"""
Linde-Buzo-Gray / Generalized Lloyd algorithm implementation in Python *3*.
Heuristic process that can be used to generate cluster points from a big amount of multidimensional vectors.
Source: https://raw.githubusercontent.com/internaut/py-lbg/master/lbg.py
"""

import math
from functools import reduce
from collections import defaultdict

_size_data = 0
_dim = 0


def generate_codebook(data, size_codebook, epsilon=0.00001):
    """
    Generate codebook of size <size_codebook> with convergence value <epsilon>. Will return a tuple with the
    generated codebook, a vector with absolute weights and a vector with relative weights (the weight denotes how many
    vectors for <data> are in the proximity of the codevector.
    :param data: input data with N k-dimensional vectors
    :param size_codebook: codebook size. Because the codevectors are split on each iteration, this should be a
                          power-of-2-value
    :param epsilon: convergence value
    :return tuple of: codebook of size <size_codebook>, absolute weights, relative weights
    """
    global _size_data, _dim

    _size_data = len(data)
    assert _size_data > 0

    _dim = len(data[0])
    assert _dim > 0

    codebook = []
    codebook_abs_weights = [_size_data]
    codebook_rel_weights = [1.0]

    # calculate initial codevector: average vector of whole input data
    c0 = avg_vec_of_vecs(data, _dim, _size_data)
    codebook.append(c0)

    # calculate the average distortion
    avg_dist = avg_distortion_c0(c0, data)

    # split codevectors until we have have enough
    while len(codebook) < size_codebook:
        codebook, codebook_abs_weights, codebook_rel_weights, avg_dist = split_codebook(data, codebook,
                                                                                        epsilon, avg_dist)

    return codebook, codebook_abs_weights, codebook_rel_weights


def split_codebook(data, codebook, epsilon, initial_avg_dist):
    """
    Split the codebook so that each codevector in the codebook is split into two.
    :param data: input data
    :param codebook: input codebook. its codevectors will be split into two.
    :param epsilon: convergence value
    :param initial_avg_dist: initial average distortion
    :return Tuple with new codebook, codebook absolute weights and codebook relative weights
    """

    # split codevectors
    new_codevectors = []
    for c in codebook:
        # the new codevectors c1 and c2 will moved by epsilon and -epsilon so to be apart from each other
        c1 = new_codevector(c, epsilon)
        c2 = new_codevector(c, -epsilon)
        new_codevectors.extend((c1, c2))

    codebook = new_codevectors
    len_codebook = len(codebook)
    abs_weights = [0] * len_codebook
    rel_weights = [0.0] * len_codebook

    # print('> splitting to size', len_codebook)

    # try to reach a convergence by minimizing the average distortion. this is done by moving the codevectors step by
    # step to the center of the points in their proximity
    avg_dist = 0
    err = epsilon + 1
    num_iter = 0
    while err > epsilon:
        # find closest codevectors for each vector in data (find the proximity of each codevector)
        closest_c_list = [None] * _size_data    # list that contains the nearest codevector for each input data vector
        vecs_near_c = defaultdict(list)         # list with codevector index -> input data vector mapping
        vec_idxs_near_c = defaultdict(list)     # list with codevector index -> input data index mapping
        for i, vec in enumerate(data):  # for each input vector
            min_dist = None
            closest_c_index = None
            for i_c, c in enumerate(codebook):  # for each codevector
                d = euclid_squared(vec, c)
                if min_dist is None or d < min_dist:    # found new closest codevector
                    min_dist = d
                    closest_c_list[i] = c
                    closest_c_index = i_c
            vecs_near_c[closest_c_index].append(vec)
            vec_idxs_near_c[closest_c_index].append(i)

        # update codebook: recalculate each codevector so that it sits in the center of the points in their proximity
        for i_c in range(len_codebook): # for each codevector index
            vecs = vecs_near_c.get(i_c) or []   # get its proximity input vectors
            num_vecs_near_c = len(vecs)
            if num_vecs_near_c > 0:
                new_c = avg_vec_of_vecs(vecs, _dim)     # calculate the new center
                codebook[i_c] = new_c                   # update in codebook
                for i in vec_idxs_near_c[i_c]:          # update in input vector index -> codevector mapping list
                    closest_c_list[i] = new_c

                # update the weights
                abs_weights[i_c] = num_vecs_near_c
                rel_weights[i_c] = num_vecs_near_c / _size_data

        # recalculate average distortion value
        prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
        avg_dist = avg_distortion_c_list(closest_c_list, data)

        # recalculate the new error value
        err = (prev_avg_dist - avg_dist) / prev_avg_dist
        # print(closest_c_list)
        # print('> iteration', num_iter, 'avg_dist', avg_dist, 'prev_avg_dist', prev_avg_dist, 'err', err)

        num_iter += 1

    return codebook, abs_weights, rel_weights, avg_dist


def avg_vec_of_vecs(vecs, dim=None, size=None):
    """
    Calculcate average vector (center vector) for input vectors <vecs>.
    :param vecs: input vectors
    :param dim: dimension of <vecs> if it was already calculated
    :param size: size of <vecs> if it was already calculated
    :return average vector (center vector) for input vectors <vecs>
    """
    size = size or len(vecs)
    dim = dim or len(vecs[0])
    avg_vec = [0.0] * dim
    for vec in vecs:
        for i, x in enumerate(vec):
            avg_vec[i] += x / size

    return avg_vec


def new_codevector(c, e):
    """
    Create a new codevector based on <c> but moved by factor <e>
    :param c: base codevector
    :param e: move factor
    :return new codevector
    """
    return [x * (1.0 + e) for x in c]


def avg_distortion_c0(c0, data, size=None):
    """
    Average distortion of <c0> in relation to <data> (i.e. how good does <c0> describe <data>?).
    :param c0: comparison vector
    :param data: sample data
    :param size: size of <data> if it was already calculated
    :return average distortion
    """
    size = size or _size_data
    return reduce(lambda s, d:  s + d / size,
                  (euclid_squared(c0, vec)
                   for vec in data),
                  0.0)


def avg_distortion_c_list(c_list, data, size=None):
    """
    Average distortion between input samples <data> and a list <c_list> that contains a codevector for each point in
    <data>.
    :param c_list: list that contains a codevector for each point in <data>
    :param data: input samples
    :param size: Size of <data> if it was already calculated
    :return:
    """
    size = size or _size_data
    return reduce(lambda s, d:  s + d / size,
                  (euclid_squared(c_i, data[i])
                   for i, c_i in enumerate(c_list)),
                  0.0)


def euclid_squared(a, b):
    return sum((x_a - x_b) ** 2 for x_a, x_b in zip(a, b))


# def euclid(a, b):
#     return math.sqrt(euclid_squared(a, b))