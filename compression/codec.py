import io
import os
import json
import numpy as np
from scipy import cluster
from collections import Counter
from pathlib import Path

from scipy.cluster.vq import vq
from skimage.measure import compare_ssim, compare_psnr

from pyfse import pyfse
from models import compression
from helpers import utils


dcn_presets = {
    '16c': './data/models/dcn/baselines/16c',
    '32c': './data/models/dcn/baselines/32c',
    '64c': './data/models/dcn/baselines/64c',
    'lq': './data/models/dcn/baselines/16c',
    'mq': './data/models/dcn/baselines/32c',
    'hq': './data/models/dcn/baselines/64c',
}


class L3ICError(Exception):
    pass


def simulate_compression(batch_x, dcn):
    """
    Simulates the entire compression and decompression (writes bytes to memory). Returns decompressed image and the byte count.
    """

    compressed_image = compress(batch_x, dcn)
    batch_y = decompress(compressed_image, dcn)

    return batch_y, len(compressed_image)


def compress_n_stats(batch_x, dcn):

    batch_y = np.zeros_like(batch_x)
    stats = {
        'ssim': np.zeros((batch_x.shape[0])),
        'psnr': np.zeros((batch_x.shape[0])),
        'entropy': np.zeros((batch_x.shape[0])),
        'bytes': np.zeros((batch_x.shape[0])),
        'bpp': np.zeros((batch_x.shape[0]))
    }

    for image_id in range(batch_x.shape[0]):
        batch_y[image_id], image_bytes = simulate_compression(batch_x[image_id:image_id + 1], dcn)
        batch_z = dcn.compress(batch_x[image_id:image_id + 1])
        stats['bytes'][image_id] = image_bytes
        stats['entropy'][image_id] = utils.entropy(batch_z, dcn.get_codebook())
        stats['ssim'][image_id] = compare_ssim(batch_x[image_id], batch_y[image_id], multichannel=True, data_range=1)
        stats['psnr'][image_id] = compare_psnr(batch_x[image_id], batch_y[image_id], data_range=1)
        stats['bpp'][image_id] = 8 * image_bytes / batch_x[image_id].shape[0] / batch_x[image_id].shape[1]

    if batch_x.shape[0] == 1:
        for k in stats.keys():
            stats[k] = stats[k][0]

    return batch_y, stats


def compare(dcn, batch_x):
    """
    Compare the quantized and decompressed image with its fully TF-processed counterpart.
    """

    code_book = dcn.sess.run(dcn.codebook).reshape((-1,))

    batch_z = dcn.compress(batch_x)
    batch_y = dcn.decompress(batch_z)

    # Quantization and coding
    indices, distortion = cluster.vq.vq(batch_z.reshape((-1)), code_book)

    # Compress each image
    data = bytes(indices.astype(np.uint8))
    coded_fse = pyfse.compress(data)
    decoded_fse = pyfse.decompress(coded_fse, int(np.prod(indices.shape)))

    # Check sanity
    assert data == decoded_fse, 'Entropy decoding error'

    shape = list(dcn.latent_shape)
    shape[0] = 1
    decoded_indices = np.array([x for x in decoded_fse]).reshape(shape)
    image_q = code_book[decoded_indices]
    image_y = dcn.decompress(image_q)

    return batch_y, image_y


def compress(batch_x, model, verbose=False):
    """
    Serialize the image as a bytes sequence. The feature maps are encoded as separate layers.

    ## Bit-stream structure:

    - Latent shape H x W x N = 3 x 1 byte (uint8)
    - Length of coded layer sizes = 2 bytes (uint16)
    - Coded layer sizes:
        - FSE encoded uint16 array of size 2 * N bytes (if possible to compress)
        - ...or RAW bytes
    - Coded layers:
        - FSE encoded uint8 array of latent vector size
        - ...or RLE encoded uint16 (number) + uint8 (byte) if all bytes are the same

    """

    if batch_x.ndim == 3:
        batch_x = np.expand_dims(batch_x, axis=0)

    assert batch_x.ndim == 4
    assert batch_x.shape[0] == 1

    image_stream = io.BytesIO()

    # Get latent space representation
    batch_z = model.compress(batch_x)
    latent_shape = np.array(batch_z.shape[1:], dtype=np.uint8)

    # Write latent space shape to the bytestream
    image_stream.write(latent_shape.tobytes())

    # Encode feature layers separately
    coded_layers = []
    code_book = model.get_codebook()
    if verbose:
        print('[l3ic encoder]', 'Code book:', code_book)

    if len(code_book) > 256:
        raise L3ICError('Code-books with more than 256 centers are not supported')

    for n in range(latent_shape[-1]):
        # TODO Should a code book always be used? What about integers?
        indices, _ = cluster.vq.vq(batch_z[:, :, :, n].reshape((-1)), code_book)

        try:
            # Compress layer with FSE
            coded_layer = pyfse.compress(bytes(indices.astype(np.uint8)))
        except pyfse.FSESymbolRepetitionError:
            # All bytes are identical, fallback to RLE
            coded_layer = np.uint16(len(indices)).tobytes() + np.uint8(indices[0]).tobytes()
        except pyfse.FSENotCompressibleError:
            # Stream does not compress
            coded_layer = np.uint8(indices).tobytes()
        finally:
            if len(coded_layer) == 1:
                if verbose:
                    layer_stats = Counter(batch_z[:, :, :, n].reshape((-1))).items()
                    print('[l3ic encoder]', 'Layer {} values:'.format(n), batch_z[:, :, :, n].reshape((-1)))
                    print('[l3ic encoder]', 'Layer {} code-book indices:'.format(n), indices.reshape((-1))[:20])
                    print('[l3ic encoder]', 'Layer {} hist:'.format(n), layer_stats)

                raise L3ICError('Layer {} data compresses to a single byte? Something is wrong!'.format(n))
            coded_layers.append(coded_layer)

    # Show example layer
    if verbose:
        n = 0
        layer_stats = Counter(batch_z[:, :, :, n].reshape((-1))).items()
        print('[l3ic encoder]', 'Layer {} values:'.format(n), batch_z[:, :, :, n].reshape((-1)))
        print('[l3ic encoder]', 'Layer {} code-book indices:'.format(n), indices.reshape((-1))[:20])
        print('[l3ic encoder]', 'Layer {} hist:'.format(n), layer_stats)

    # Write the layer size array
    layer_lengths = np.array([len(x) for x in coded_layers], dtype=np.uint16)

    try:
        coded_lengths = pyfse.compress(layer_lengths.tobytes())
        if verbose: print('[l3ic encoder]', 'FSE coded lengths')
    except pyfse.FSENotCompressibleError:
        # If the FSE coded stream is empty - it is not compressible - save natively
        if verbose: print('[l3ic encoder]', 'RAW coded lengths')
        coded_lengths = layer_lengths.tobytes()

    if verbose:
        print('[l3ic encoder]', 'Coded lengths #', len(coded_lengths), '=', coded_lengths)
        print('[l3ic encoder]', 'Layer lengths = ', layer_lengths)

    if len(coded_lengths) == 0:
        raise RuntimeError('Empty coded layer lengths!')

    image_stream.write(np.uint16(len(coded_lengths)).tobytes())
    image_stream.write(coded_lengths)

    # Write individual layers
    for layer in coded_layers:
        image_stream.write(layer)

    return image_stream.getvalue()


def decompress(stream, model=None, verbose=False):
    """
    Decompress an image from the given bytes sequence. See docs of compress for stream details.
    """

    if type(stream) is bytes:
        stream = io.BytesIO(stream)
    elif type(stream) is io.BytesIO:
        pass
    elif not hasattr(stream, 'read'):
        raise ValueError('Unsupported stream type!')

    # Read the shape of the latent representation
    latent_x, latent_y, n_latent = np.frombuffer(stream.read(3), np.uint8)

    # Read the array with layer sizes
    layer_bytes = np.frombuffer(stream.read(2), np.uint16)
    coded_layer_lengths = stream.read(int(layer_bytes))

    if verbose:
        print('[l3ic decoder]', 'Latent space', latent_x, latent_y, n_latent)
        print('[l3ic decoder]', 'Layer bytes', layer_bytes)

    if layer_bytes != 2 * n_latent:
        if verbose:
            print('[l3ic decoder]', 'Decoding FSE L')
            print('[l3ic decoder]', 'Decoding from', coded_layer_lengths)
        layer_lengths_bytes = pyfse.decompress(coded_layer_lengths)
        layer_lengths = np.frombuffer(layer_lengths_bytes, dtype=np.uint16)
    else:
        if verbose:
            print('[l3ic decoder]', 'Decoding RAW L')
        layer_lengths = np.frombuffer(coded_layer_lengths, dtype=np.uint16)

    if verbose:
        print('[l3ic decoder]', 'Layer lengths', layer_lengths)

    # Get the correct DCN model
    if model is None:
        model = compression.DCN('{}c'.format(n_latent))

    if model.n_latent != n_latent:
        print('[l3ic decoder]', 'WARNING', 'the specified model ({}c) does not match the coded stream ({}c) - switching'.format(model.n_latent, n_latent))
        model = compression.DCN('{}c'.format(n_latent))

    code_book = model.get_codebook()
    
    # Create the latent space array
    batch_z = np.zeros((1, latent_x, latent_y, n_latent))

    # Decompress the features separately
    for n in range(n_latent):
        coded_layer = stream.read(int(layer_lengths[n]))
        try:
            if len(coded_layer) == 3:
                # RLE encoding
                count = np.frombuffer(coded_layer[:2], dtype=np.uint16)[0]
                layer_data = coded_layer[-1:] * int(count)
            elif len(coded_layer) == int(latent_x) * int(latent_y):
                # If the data could not have been compressed, just read the raw stream
                layer_data = coded_layer
            else:
                layer_data = pyfse.decompress(coded_layer, 4 * latent_x * latent_y)
        except pyfse.FSEException as e:
            print('[l3ic decoder]', 'ERROR while decoding layer', n)
            print('[l3ic decoder]', 'Stream of size', len(coded_layer), 'bytes =', coded_layer)
            raise e
        batch_z[0, :, :, n] = code_book[np.frombuffer(layer_data, np.uint8)].reshape((latent_x, latent_y))

    # Show example layer
    if verbose:
        n = 0
        layer_stats = Counter(batch_z[:, :, :, n].reshape((-1))).items()
        print('[l3ic decoder]', 'Layer {} values:'.format(n), batch_z[:, :, :, n].reshape((-1)))
        print('[l3ic decoder]', 'Layer {} hist:'.format(n), layer_stats)

    # Use the DCN decoder to decompress the RGB image
    return model.decompress(batch_z)


def global_compress(dcn, batch_x):
    # Naive FSE compression of the entire latent repr.
    batch_z = dcn.compress(batch_x)
    indices, distortion = vq(batch_z.reshape((-1)), dcn.get_codebook())
    return pyfse.compress(bytes(indices.astype(np.uint8)))


def restore_model(dir_name, patch_size=128, fetch_stats=False, sess=None, graph=None, x=None, nip_input=None):
    """
    Utility function to restore a DCN model from a training directory. By default,
    a standalone instance is created. Can also be used for chaining when sess,
    graph, x, nip_input are provided.

    :param dir_name: directory with a trained model (with progress.json)
    :param patch_size: input patch size (scalar)
    :param fetch_stats: return a tuple (model, training_stats)
    :param sess: existing TF session of None
    :param graph: existing TF graph or None
    :param x: input to the model
    :param nip_input: input to the NIP model (useful for chaining)
    """
    training_progress_path = None

    if dir_name in dcn_presets:
        dir_name = dcn_presets[dir_name]

    if dir_name is None:
        raise ValueError('dcn directory cannot be None')

    if not os.path.exists(dir_name):
        raise ValueError('Directory {} does not exist!'.format(dir_name))

    for filename in Path(dir_name).glob('**/progress.json'):
        training_progress_path = str(filename)

    if training_progress_path is None:
        raise FileNotFoundError('Could not find a DCN model snapshot (json+checkpoint) in {}'.format(dir_name))

    with open(training_progress_path) as f:
        training_progress = json.load(f)

    parameters = training_progress['dcn']['args']
    parameters['patch_size'] = patch_size
    parameters['default_val_is_train'] = False

    if x is not None:
        parameters['x'] = x
    if nip_input is not None:
        parameters['nip_input'] = nip_input

    model = getattr(compression, training_progress['dcn']['model'])(sess, graph, **parameters)
    model.load_model(dir_name)
    print('Loaded model: {}'.format(model.model_code))

    if fetch_stats:

        # TODO Entropy is fetched from training measurements instead of validation (didn't get recorded)
        stats = {
            'loss': np.round(training_progress['performance']['loss']['validation'][-1], 3),
            'entropy': np.round(training_progress['performance']['entropy']['training'][-1], 3),
            'ssim': np.round(training_progress['performance']['ssim']['validation'][-1], 3)
        }

        return model, stats
    else:
        return model
