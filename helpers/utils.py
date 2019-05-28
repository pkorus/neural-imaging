import io
import imageio
import numpy as np
import scipy.stats as st

_numeric_types = {int, float, bool, np.bool, np.float, np.float16, np.float32, np.float64,
                           np.int, np.int8, np.int32, np.int16, np.int64,
                           np.uint, np.uint8, np.uint32, np.uint16, np.uint64}


def ma_gaussian(x, y, step_size=0.05, width=10):
    """Moving average with Gaussian averaging"""
    bin_centers = np.arange(np.min(x), np.max(x) - 0.5*step_size, step_size) + 0.5*step_size
    bin_avg = np.zeros(len(bin_centers))

    # We're going to weight with a Gaussian function
    def gaussian(x, amp=1, mean=0, sigma=1):
        return amp*np.exp(-(x-mean)**2/(2*sigma**2))

    for index in range(0,len(bin_centers)):
        bin_center = bin_centers[index]
        weights = gaussian(x, mean=bin_center, sigma=width)
        bin_avg[index] = np.average(y, weights=weights)

    return bin_centers, bin_avg


def ma_conv(x, n=10):
    """Moving average with simple box filter."""

    if len(x) == 0:
        return np.array([])

    if n == 0:
        n = (len(x) // 10)

    fn = 2*n + 1

    return np.convolve(np.pad(x, n, 'edge'), np.ones((fn,))/fn, mode='valid')


def normalize(x, perc=0):
    """
    Normalize the input array to [0, 1]. Optionally, cut top and bottom outliers (based on percentiles).
    """
    if perc == 0:
        return ((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)).clip(0, 1)
    else:
        mn = np.percentile(x, perc)
        mx = np.percentile(x, 100 - perc)
        return ((x - mn) / (mx - mn + 1e-9)).clip(0, 1)


def split_dataset(files, n_test=0.1, n_valid=0.1):
    """
    Split a list of files into training, testing and validation sets.
    """
    n_images = len(files)
    n_test = n_test if n_test > 1 else np.floor(n_images * n_test)
    n_valid = n_valid if n_valid > 1 else np.floor(n_images * n_valid)
    n_train = n_images - n_test - n_valid
    
    randomized_indices = np.random.permutation(n_images)
    
    files = np.array(files)
    
    files_train = files[randomized_indices[0:n_train]]
    files_test = files[randomized_indices[n_train:n_train+n_test]]
    files_valid = files[randomized_indices[-n_valid:]]
    
    # Sanity check
    if len([x for x in files_test if x in files_train]) > 0:
        raise ValueError('Test file also in training set!')

    if len([x for x in files_valid if x in files_train]) > 0:
        raise ValueError('Validation file also in training set!')
        
    return files_train.tolist(), files_test.tolist(), files_valid.tolist()


def stack_bayer(image_rgb, cfa_pattern):
    """
    Return a RGGB Bayer stack sampled from a RGB image according to a given CFA configuration.
    :param image_rgb: 3-D numpy array (h, w, 3:rgb)
    :param cfa_pattern: 'GBRG', 'RGGB' or 'BGGR'
    """

    if cfa_pattern.upper() == 'GBRG':
        r = image_rgb[1::2, 0::2, 0]
        g1 = image_rgb[0::2, 0::2, 1]
        g2 = image_rgb[1::2, 1::2, 1]
        b = image_rgb[0::2, 1::2, 2]
        
    elif cfa_pattern.upper() == 'RGGB':
        r = image_rgb[0::2, 0::2, 0]
        g1 = image_rgb[0::2, 1::2, 1]
        g2 = image_rgb[1::2, 0::2, 1]
        b = image_rgb[1::2, 1::2, 2]
        
    elif cfa_pattern.upper() == 'BGGR':
        r = image_rgb[1::2, 1::2, 0]
        g1 = image_rgb[0::2, 1::2, 1]
        g2 = image_rgb[1::2, 0::2, 1]
        b = image_rgb[0::2, 0::2, 1]
    
    return np.dstack([r, g1, g2, b])


def merge_bayer(bayer_stack, cfa_pattern):
    """
    Merge a RGGB Bayer stack into a RGB image.
    :param bayer_stack: a 3-D numpy array (h/2, w/2, 4:rggb)
    :param cfa_pattern: 'GBRG', 'RGGB' or 'BGGR'
    """
    if bayer_stack.ndim == 4:
        
        if bayer_stack.shape[0] != 1:
            raise ValueError('4-D arrays are not supported!')
        
        bayer_stack = bayer_stack[0, :, :, :]
    
    assert bayer_stack.ndim == 3
    
    H = bayer_stack.shape[0]
    W = bayer_stack.shape[1]
    
    image_rgb = np.zeros((2*H, 2*W, 3), dtype=bayer_stack.dtype)
        
    if cfa_pattern == 'GBRG':    
        image_rgb[1::2, 0::2, 0] = bayer_stack[:, :, 0]
        image_rgb[0::2, 0::2, 1] = bayer_stack[:, :, 1]
        image_rgb[1::2, 1::2, 1] = bayer_stack[:, :, 2]
        image_rgb[0::2, 1::2, 2] = bayer_stack[:, :, 3]
        
    elif cfa_pattern == 'RGGB':    
        image_rgb[0::2, 0::2, 0] = bayer_stack[:, :, 0]
        image_rgb[0::2, 1::2, 1] = bayer_stack[:, :, 1]
        image_rgb[1::2, 0::2, 1] = bayer_stack[:, :, 2]
        image_rgb[1::2, 1::2, 2] = bayer_stack[:, :, 3]
        
    elif cfa_pattern == 'BGGR':    
        image_rgb[1::2, 1::2, 0] = bayer_stack[:, :, 0]
        image_rgb[0::2, 1::2, 1] = bayer_stack[:, :, 1]
        image_rgb[1::2, 0::2, 1] = bayer_stack[:, :, 2]
        image_rgb[0::2, 0::2, 1] = bayer_stack[:, :, 3]
    
    return image_rgb


def upsampling_kernel(cfa_pattern='gbrg'):
    """
    Possible initializations of up-sampling kernels for matching the 12-feature-layer format needed by depth-to-space.
    (Ideally, this should match the CFA pattern of the camera).
    :param cfa_pattern: 'GBRG'
    """

    # TODO Implement other CFA patterns
    if cfa_pattern.upper() == 'GBRG':
        #                R  G  B  R  G  B  R  G  B  R  G  B
        #                1  1  1  2  2  2  3  3  3  4  4  4
        upk = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],                 
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                       ])
    else:
        raise ValueError('Unsupported CFA pattern: {}'.format(cfa_pattern))
        
    return upk


def gamma_kernels():
    """
    Pre-trained kernels of a toy neural network for approximation of gamma correction.
    """
    gamma_dense1_kernel = np.array([2.9542332, 17.780445, 0.6280197, 0.40384966])
    gamma_dense1_bias = np.array([0.4047071, 1.1489044, -0.17624384, 0.47826886])

    gamma_dense2_kernel = np.array([0.44949612, 0.78081024, 0.97692937, -0.24265033])
    gamma_dense2_bias = np.array([-0.4702738])

    gamma_d1k = np.zeros((3, 12))
    gamma_d1b = np.zeros((12, ))
    gamma_d2k = np.zeros((12, 3))
    gamma_d2b = np.zeros((3,))

    for r in range(3):
        gamma_d1k[r, r*4:r*4+4] = gamma_dense1_kernel
        gamma_d1b[r*4:r*4+4] = gamma_dense1_bias
        gamma_d2k[r*4:r*4+4, r] = gamma_dense2_kernel
        gamma_d2b[r] = gamma_dense2_bias
    
    return gamma_d1k, gamma_d1b, gamma_d2k, gamma_d2b


def bilin_kernel(kernel=3):
    """
    Bilinear demosaicing kernels.
    """
    g_kern = np.array([[0, 1/4., 0], [1/4., 1, 1/4.], [0, 1/4., 0]])
    rb_kern = np.array([[1/4., 1/2., 1/4.], [1/2., 1, 1/2.], [1/4., 1/2., 1/4.]])

    G_kern = np.zeros((3,3,3), np.float32)
    G_kern[:, :, 1] = g_kern

    R_kern = np.zeros((3,3,3), np.float32)
    R_kern[:, :, 0] = rb_kern

    B_kern = np.zeros((3,3,3), np.float32)
    B_kern[:, :, 2] = rb_kern

    dmf = np.stack((R_kern, G_kern, B_kern), axis=3)
    if kernel > 3:         
        pad = (kernel - 3) // 2
        dmf = np.pad(dmf, ((pad, pad), (pad, pad), (0, 0), (0, 0)), 'constant', constant_values=0)
        
    return dmf


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig + 1.) / kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    return kernel


def jpeg_qtable(quality, channel=0):
    """
    Return a DCT quantization matrix for a given quality level.
    :param quality: JPEG quality level (1-100)
    :param channel: 0 for luminance, >0 for chrominance channels
    """

    # Sanitize
    quality = np.maximum(np.minimum(100, quality), 1)
    
    # Convert to linear quality scale
    quality = 5000 / quality if quality < 50 else 200 - quality*2
    
    if channel == 0:
        # This is table 0 (the luminance table):
        t = [[16,  11,  10,  16,  24,  40,  51,  61], 
             [12,  12,  14,  19,  26,  58,  60,  55],
             [14,  13,  16,  24,  40,  57,  69,  56],
             [14,  17,  22,  29,  51,  87,  80,  62],
             [18,  22,  37,  56,  68, 109, 103,  77],
             [24,  35,  55,  64,  81, 104, 113,  92],
             [49,  64,  78,  87, 103, 121, 120, 101],
             [72,  92,  95,  98, 112, 100, 103,  99]]
        t = np.array(t, np.float32)
    
    else:
        # This is table 1 (the chrominance table):
        t = [[17,  18,  24,  47,  99,  99,  99,  99],
             [18,  21,  26,  66,  99,  99,  99,  99],
             [24,  26,  56,  99,  99,  99,  99,  99],
             [47,  66,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99]]
        t = np.array(t, np.float32)    
    
    t = np.floor((t * quality + 50)/100)    
    t[t < 1] = 1    
    t[t > 255] = 255

    return t


def repeat_2dfilter(f, channels):
    rf = np.zeros((f.shape[0], f.shape[1], channels, channels))

    for r in range(channels):
        rf[:, :, r, r] = f

    return rf


def center_mask_2dfilter(f_size, channels):
    indicator = np.zeros((f_size, f_size, channels, channels))

    for r in range(channels):
        indicator[f_size // 2, f_size // 2, r, r] = 1

    return indicator


def slidingwindow(arr, window):
    if arr.ndim != 3:
        raise ValueError('The input array needs to be 3-D - (h,w,c)!')
    n_windows = (arr.shape[0] // window) * (arr.shape[1] // window)
    batch = np.zeros((n_windows, window, window, arr.shape[-1]), dtype=arr.dtype)
    window_id = 0
    for x in range(arr.shape[1] // window):
        for y in range(arr.shape[0] // window):
            batch[window_id] = arr[y*window:(y+1)*window, x*window:(x+1)*window, :]
            window_id += 1
    return batch


def is_number(value):
    return type(value) in _numeric_types


def is_numeric_type(t):
    return t in _numeric_types


def is_nan(value):

    if value is None:
        return True

    if is_number(value):
        return np.isnan(value)

    return False

