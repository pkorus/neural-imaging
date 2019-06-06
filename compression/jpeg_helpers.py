import io
import numpy as np
import imageio
import scipy as sp
from skimage.measure import compare_ssim
from collections import OrderedDict
from struct import unpack

app_markers = (0xffe0, 0xffe1, 0xffe2, 0xffe3, 0xffe4, 0xffe5, 0xffe6, 0xffe7, 0xffe8, 0xffe9, 0xffea, 0xffeb, 0xffec,
               0xffed, 0xffee, 0xffef)

markers = {
    'SOS': 'Start of Stream',
    'SOI': 'Start of Image',
    'APP': 'Application',
    'EOI': 'End of Image',
    'DHT': 'Huffman tables',
    'DQT': 'DCT Quantization Tables',
    'ECD': 'Entropy Coded Data'
}


def match_ssim(image, ssim=0.95, subsampling='4:4:4'):

    assert image.ndim == 3, 'Only RGB images supported'

    def fun(q):
        image_j = compress_batch(image, q, subsampling=subsampling)[0].squeeze()
        c_ssim = compare_ssim(image, image_j, multichannel=True, data_range=1)
        return c_ssim - ssim

    low = 1
    high = 95
    low_ssim = fun(low)
    high_ssim = fun(high)

    while True:

        if high - low <= 1:
            if abs(high_ssim) > abs(low_ssim):
                return low
            else:
                return high

        if (low_ssim) * (high_ssim) > 0:
            raise ValueError('Same deviation for both end-points')

        mid = int((low + high)/2)
        mid_ssim = fun(mid)

        if (mid_ssim) * (high_ssim) > 0:
            high = mid
            high_ssim = mid_ssim
        else:
            low = mid
            low_ssim = mid_ssim


def compress_batch(batch_x, jpeg_quality, effective=False, subsampling='4:4:4'):

    if batch_x.max() > 1:
        batch_x = batch_x.astype(np.float32) / (2**8 - 1)

    if batch_x.ndim == 3:
        s = io.BytesIO()
        imageio.imsave(s, (255 * batch_x).astype(np.uint8).squeeze(), format='jpg', quality=jpeg_quality, subsampling=subsampling)
        image_compressed = imageio.imread(s.getvalue())
        image_bytes = len(s.getvalue()) if not effective else JPEGStats(s.getvalue()).get_effective_bytes()

        return image_compressed / (2 ** 8 - 1), image_bytes

    elif batch_x.ndim == 4:
        batch_j = np.zeros_like(batch_x)
        bytes_arr = []
        for r in range(batch_x.shape[0]):
            s = io.BytesIO()
            imageio.imsave(s, (255 * batch_x[r]).astype(np.uint8).squeeze(), format='jpg', quality=jpeg_quality, subsampling=subsampling)
            image_compressed = imageio.imread(s.getvalue())
            batch_j[r] = image_compressed.astype(np.float32) / (2 ** 8 - 1)
            image_bytes = len(s.getvalue()) if not effective else JPEGStats(s.getvalue()).get_effective_bytes()
            bytes_arr.append(image_bytes)

        return batch_j, bytes_arr


def get_byte_array(chunk):
    """ convert chunk of bytes to corresponding byte array"""
    return list(unpack("B" * len(chunk), chunk))


class JPEGStats:
    """
    Object to calculate various block starts from encoded JPEG data
    Keyword arguments:
        - l_decode  : length of part decoded till now
        - len_chunk : length of the current block
        - blocks    : dictionary of block names to their starting index
    """

    def __init__(self, image):
        self.l_decode = 0
        self.len_chunk = 0
        self.blocks = OrderedDict()

        if type(image) is str:
            with open(image, 'rb') as f:
                image = f.read()
        elif type(image) is bytes:
            pass
        else:
            raise ValueError('Image not supported! Supported: str, bytes')

        self._process(image)
        self.image = imageio.imread(image)

    def _process_quantization_tables(self, data):
        """ extracts the quantization table from data and updates the JPEG object"""
        while len(data) > 0:
            # get the ID of the table [Luma(0), Chroma(1)]
            marker, = unpack("B", data[0:1])
            # get the complete table of 64 elements in one go
            # self.blocks['quant_{}'.format(marker & 0xf)] = self.l_decode
            self.blocks['DQT:{}'.format(marker & 0xf)] = self.l_decode
            # remove the quantization table chunk
            data = data[65:]

    def _process_huffman_tables(self, data):
        """ extracts the Huffman tables from data, creates an HuffmanTable object and updates the JPEG object"""
        while len(data) > 0:
            id, = unpack("B", data[0: 1])
            lengths = get_byte_array(data[1: 17])
            data = data[17:]
            for i in lengths:
                data = data[i:]
            self.blocks['DHT:{}'.format(id)] = self.l_decode

    def _process(self, data):
        """ Use in-class functions to decode the binary data and return its length on success. """
        temp_data = data
        rst_marker_index = 0
        app_marker_index = 0
        self.blocks['SOI'] = 0
        try:
            while len(data) > 0:
                # unpacking big endian hexadecimal marker
                marker, = unpack(">H", data[0:2])
                # start of image
                if marker == 0xffd8:
                    self.len_chunk = 2
                    self.l_decode = 2
                # end of image
                elif marker == 0xffd9:
                    self.l_decode += 2
                    self.blocks['EOI'] = self.l_decode
                    return self.blocks
                else:
                    # decode the image chunk by chunk
                    # getting the bytes denoting length
                    self.len_chunk, = unpack(">H", data[2:4])
                    # add the length of the 2 bytes marker
                    self.len_chunk += 2
                    # get the chunk after removing marker and length bytes
                    chunk = data[4:self.len_chunk]
                    if marker == 0xffdb:
                        self._process_quantization_tables(chunk)
                    elif marker == 0xffc0:
                        self.blocks['DCT'] = self.l_decode
                    elif marker == 0xffc2:
                        print("Skipping progressive mode fragments")
                        raise NotImplementedError('Progressive JPEG images not supported yet')
                    elif marker == 0xffc4:
                        self._process_huffman_tables(chunk)
                    elif marker == 0xffda:
                        # assuming valid JPEG
                        self.blocks['SOS'] = self.l_decode
                        self.len_chunk, = unpack(">H", data[2:4])
                        self.len_chunk += 2
                        self.l_decode += self.len_chunk
                        data = data[self.len_chunk:]
                        self.len_chunk = len(temp_data) - self.l_decode - 2
                        self.blocks['ECD'] = self.l_decode
                    elif marker in app_markers:
                        # suppose header contains two app markers then for ex, ffed -> app_13_0 and ffe0 -> app_0_1
                        # self.blocks['app_{}_{}'.format(0xf & marker, app_marker_index)] = self.l_decode
                        self.blocks['APP:{}/{}'.format(0xf & marker, app_marker_index)] = self.l_decode
                        app_marker_index += 1
                    elif marker in (0xfffe, 0xffdd):
                        # self.blocks['rst_{}'.format(rst_marker_index)] = self.l_decode
                        self.blocks['RST'] = self.l_decode
                        rst_marker_index += 1
                    else:
                        break
                    self.l_decode += self.len_chunk
                data = data[self.len_chunk:]
        except Exception as e:
            raise IOError('Parsing error: {}'.format(e))

        return None

    def get_bytes(self):
        return self.blocks['EOI']

    def get_effective_bytes(self):
        return self.blocks['EOI'] - self.blocks['DHT:0']

    def get_effective_bpp(self):
        return 8 * self.get_effective_bytes() / self.image.shape[0] / self.image.shape[1]

    def get_bpp(self):
        return 8 * self.blocks['EOI'] / self.image.shape[0] / self.image.shape[1]
