"""
These functions have been adopted from https://github.com/fab-jul/imgcomp-cvpr
"""
import os
import subprocess


_BPG_QUANTIZATION_PARAMETER_RANGE = (1, 51)  # smaller means better
BPGENC = os.environ.get('BPGENC', 'bpgenc')


def bpg_compress(input_image_p, q, tmp_dir=None, chroma_fmt='444'):
    """ Int -> image_out_path :: str """
    assert 'png' in input_image_p
    if tmp_dir:
        input_image_name = os.path.basename(input_image_p)
        output_image_bpg_p = os.path.join(tmp_dir, input_image_name).replace('.png', '_tmp_bpg.bpg')
    else:
        output_image_bpg_p = input_image_p.replace('.png', '_tmp_bpg.bpg')
    subprocess.call([BPGENC, '-q', str(q), input_image_p, '-o', output_image_bpg_p, '-f', chroma_fmt])
    return output_image_bpg_p


def decode_bpg_to_png(bpg_p):  # really fast
    png_p = bpg_p.replace('.bpg', '_as_png.png')
    subprocess.call(['bpgdec', '-o', png_p, bpg_p])
    return png_p


def bpp_of_bpg_image(bpg_p):
    return bpg_image_info(bpg_p).bpp


class BPGImageInfo(object):
    def __init__(self, width, height, num_bytes_for_picture):
        self.width = width
        self.height = height
        self.num_bytes_for_picture = num_bytes_for_picture
        self.bpp = num_bytes_for_picture * 8 / float(width * height)


def bpg_image_info(p):
    """
    Relevant format spec:
    magic number          4 bytes
    header stuff          2 bytes
    width                 variable, ue7
    height                variable, ue7
    picture_data_length   variable, ue7. If zero: remaining data is image
    """
    with open(p, 'rb') as f:
        magic = f.read(4)
        expected_magic = bytearray.fromhex('425047fb')
        assert magic == expected_magic, 'Not a BPG file it seems: {}'.format(p)
        header_info = f.read(2)
        width = _read_ue7(f)
        height = _read_ue7(f)
        picture_data_length = _read_ue7(f)
        num_bytes_for_picture = _number_of_bytes_until_eof(f) if picture_data_length == 0 else picture_data_length
        return BPGImageInfo(width, height, num_bytes_for_picture)


def _read_ue7(f):
    """
    ue7 means it's a bunch of bytes all starting with a 1 until one byte starts
    with 0. from all those bytes you take all bits except the first one and
    merge them. E.G.

    some ue7-encoded number:      10001001 01000010
    take all bits except first ->  0001001  1000010 
    merge ->                            10011000010 = 1218
    """
    bits = 0
    first_bit_mask = 1 << 7
    value_holding_bits_mask = int(7 * '1', 2) 
    for byte in _byte_generator(f):
        byte_as_int = byte[0]
        more_bits_are_coming = byte_as_int & first_bit_mask
        bits_from_this_byte = byte_as_int & value_holding_bits_mask
        bits = (bits << 7) | bits_from_this_byte
        if not more_bits_are_coming:
            return bits


def _number_of_bytes_until_eof(f):
    return sum(1 for _ in _byte_generator(f))


def _byte_generator(f):
    while True:
        byte = f.read(1)
        if byte == b"":
            break
        yield byte
