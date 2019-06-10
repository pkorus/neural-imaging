import numpy as np
from scipy import cluster

from pyfse import pyfse


def dcn_simulate_compression(dcn, batch_x):

    code_book = dcn.sess.run(dcn.codebook).reshape((-1,))

    batch_z = dcn.compress(batch_x)
    batch_y = dcn.decompress(batch_z)

    # Quantization and coding
    indices, distortion = cluster.vq.vq(batch_z.reshape((-1)), code_book)

    # Compress each image
    data = bytes(indices.astype(np.uint8))
    coded_fse = pyfse.easy_compress(data)
    decoded_fse = pyfse.easy_decompress(coded_fse, int(np.prod(indices.shape)))
    image_bytes = len(coded_fse)

    # Check sanity
    assert data == decoded_fse, 'Entropy decoding error'

    shape = list(dcn.latent_shape)
    shape[0] = 1
    decoded_indices = np.array([x for x in decoded_fse]).reshape(shape)
    image_q = code_book[decoded_indices]
    image_y = dcn.decompress(image_q)

    if not np.all(np.abs((255*batch_y[0]).astype(np.uint8) - (255*image_y).astype(np.uint8)) <= 1):
        print('WARNING De-compressed image seems to be different from the simulated one!')

    return batch_y, image_bytes


def dcn_compare(dcn, batch_x):

    code_book = dcn.sess.run(dcn.codebook).reshape((-1,))

    batch_z = dcn.compress(batch_x)
    batch_y = dcn.decompress(batch_z)

    # Quantization and coding
    indices, distortion = cluster.vq.vq(batch_z.reshape((-1)), code_book)

    # Compress each image
    data = bytes(indices.astype(np.uint8))
    coded_fse = pyfse.easy_compress(data)
    decoded_fse = pyfse.easy_decompress(coded_fse, int(np.prod(indices.shape)))

    # Check sanity
    assert data == decoded_fse, 'Entropy decoding error'

    shape = list(dcn.latent_shape)
    shape[0] = 1
    decoded_indices = np.array([x for x in decoded_fse]).reshape(shape)
    image_q = code_book[decoded_indices]
    image_y = dcn.decompress(image_q)

    return batch_y, image_y