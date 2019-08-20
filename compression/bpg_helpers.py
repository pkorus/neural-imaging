import os
import subprocess
import operator


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


def bpg_measure(input_image_p, bpp, precise=False, save_output_as_png=None, tmp_dir=None):
    """
    :return (PSNR, SSIM, MS-SSIM, actual_bpp)
    """
    input_image_root_p, ext = os.path.splitext(input_image_p)
    assert ext == '.png', 'Expected PNG to convert to BMP, got {}'.format(input_image_p)
    output_image_bpg_p, actual_bpp = _bpg_compress_to_bpp(
        input_image_p, target_bpp=bpp, precise=precise)
    output_image_bpg_png_p = decode_bpg_to_png(output_image_bpg_p)
    os.remove(output_image_bpg_p)  # don't need that anymore

    _, msssim, _ = compare_imgs.compare(input_image_p, output_image_bpg_png_p,
                                        calc_ssim=False, calc_msssim=True, calc_psnr=False)
    if save_output_as_png:
        os.rename(output_image_bpg_png_p, save_output_as_png)
    else:
        os.remove(output_image_bpg_png_p)
    return msssim, actual_bpp


def _bpg_compress_to_bpp(input_image_p, target_bpp, precise=False, tmp_dir=None):
    def compress_input_image_with_quality(q):
        return bpg_compress(input_image_p, q, tmp_dir)

    bpp_eps = 0.01 if precise else 0.05
    try:
        q_min, q_max = _BPG_QUANTIZATION_PARAMETER_RANGE
        output_image_bpg_p, q = binary_search(compress_input_image_with_quality, bpp_of_bpg_image, 'decreasing',
                                              y_target=target_bpp, y_target_eps=bpp_eps,
                                              x_min=q_min, x_max=q_max, x_eps=0.1, log=False)
    except BinarySearchFailedException as e:
        q = e.first_x_yielding_y_greater_than(target_bpp)
        output_image_bpg_p = compress_input_image_with_quality(q)

    print('q = {}'.format(q))
    actual_bpp = bpp_of_bpg_image(output_image_bpg_p)
    return output_image_bpg_p, actual_bpp


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


class BinarySearchFailedException(Exception):
    def __init__(self, discovered_values):
        self.discovered_values = discovered_values

    def first_x_yielding_y_greater_than(self, y_target):
        for x, y in sorted(self.discovered_values, key=operator.itemgetter(1)):
            if y > y_target:
                return x
        raise ValueError('No x found with y > {} in {}.'.format(y_target, self.discovered_values))


def binary_search(f, g, f_type, y_target, y_target_eps, x_min, x_max, x_eps, max_num_iter=1000, log=True):
    """ does binary search on f :: X -> Z by calculating z = f(x) and using g :: Z -> Y to get y = g(z) = g(f(x)).
    (g . f) is assumed to be monotonically increasing iff f_tpye == 'increasing' and monotonically decreasing iff
    f_type == 'decreasing'.
    Returns first (x, z) for which |y_target - g(f(x))| < y_target_eps. x_min, x_max specifiy initial search interval for x.
    Stops if x_max - x_min < x_eps. Raises BinarySearchFailedException when x interval too small or if search takes
    more than max_num_iter iterations. The expection has a field `discovered_values` which is a list of checked
    (x, y) coordinates. """
    def _print(s):
        if log:
            print(s)
    assert f_type in ('increasing', 'decreasing')
    cmp_op = operator.gt if f_type == 'increasing' else operator.lt
    discovered_values = []
    print_col_width = len(str(x_max)) + 3
    for _ in range(max_num_iter):
        x = x_min + (x_max - x_min) / 2
        z = f(x)
        y = g(z)
        discovered_values.append((x, y))
        _print('[{:{width}.2f}, {:{width}.2f}] -- g(f({:{width}.2f})) = {:.2f}'.format(
            x_min, x_max, x, y, width=print_col_width))
        if abs(y_target - y) < y_target_eps:
            return z, x
        if cmp_op(y, y_target):
            x_max = x
        else:
            x_min = x
        if x_max - x_min < x_eps:
            _print('Stopping, interval too close!')
            break
    sorted_discovered_values = sorted(discovered_values)
    first_y, last_y = sorted_discovered_values[0][1], sorted_discovered_values[-1][1]
    if (f_type == 'increasing' and first_y > last_y) or (f_type == 'decreasing' and first_y < last_y):
        raise ValueError('Got f_type == {}, but first_y, last_y = {}, {}'.format(
            f_type, first_y, last_y))
    raise BinarySearchFailedException(discovered_values)

