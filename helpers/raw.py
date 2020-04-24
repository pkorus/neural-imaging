# -*- coding: utf-8 -*-
"""
Helper functions for developing RAW images and working with Bayer patterns.
"""
import colour_demosaicing
from rawkit.raw import Raw
import numpy as np


def process(filename, use_srgb=True, use_gamma=True, brightness='percentile', demosaicing='menon', expand=False):
    """
    A simple imaging pipeline implemented from scratch.
    :param filename: input RAW image
    :param use_srgb: set to False to disable camera RGB to sRGB conversion
    :param use_gamma: set to False to disable gamma correction
    :param brightness: global brightness correction method (percentile, shift or None)
    :param demosaicing: demosaicing method (menon, bilinear)
    """

    # Sanity checks
    if brightness not in ['percentile', 'shift', None]:
        raise ValueError('Unsupported brightness correction mode!')
        
    if demosaicing not in ['menon', 'bilinear']:
        raise ValueError('Unsupported demosaicing method!')
    
    with Raw(filename) as raw:
        raw.unpack()
        image_raw = np.array(raw.raw_image(), dtype=np.float32)

        # Normalization and calibration
        black = raw.data.contents.color.black
        saturation = raw.data.contents.color.maximum

        image_raw = image_raw.astype(np.float32)
        image_raw -= black
        
        uint14_max = 1
        image_raw *= uint14_max / (saturation - black)
        image_raw = np.clip(image_raw, 0, uint14_max)
            
        # White balancing
        cam_mul = np.array(raw.data.contents.color.cam_mul, dtype=np.float32)
        cam_mul /= cam_mul[1] # Set the multiplier for G to be 1
        
        cfa_pattern = ''.join([''.join(x) for x in raw.color_filter_array])
        
        if cfa_pattern == 'GBRG':    
            image_raw[1::2, 0::2] *= cam_mul[0]
            image_raw[0::2, 1::2] *= cam_mul[2]
        elif cfa_pattern == 'RGGB':    
            image_raw[0::2, 0::2] *= cam_mul[0]
            image_raw[1::2, 1::2] *= cam_mul[2]
        elif cfa_pattern == 'BGGR':    
            image_raw[1::2, 1::2] *= cam_mul[0]
            image_raw[0::2, 0::2] *= cam_mul[2]        
            
        image_raw = image_raw.clip(0, uint14_max)
        
        # Demosaicing
        if demosaicing == 'menon':
            image_rgb = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(image_raw, pattern=cfa_pattern)
        elif demosaicing == 'bilinear':
            image_rgb = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(image_raw, pattern=cfa_pattern)
            
        # Color space conversion
        if use_srgb:
            cam2srgb = np.array(raw.data.contents.color.rgb_cam, dtype=np.float).reshape((3,4))[:, 0:3]
            
            shape = image_rgb.shape
            pixels = image_rgb.reshape(-1, 3).T
            pixels = cam2srgb.dot(pixels)
            
            image_rgb = pixels.T.reshape(shape)
            image_rgb = image_rgb.clip(0, uint14_max)
            
            # Deallocate
            del pixels
        
        # Brightness correction
        if brightness == 'percentile':
            percentile = 0.5
            image_rgb -= np.percentile(image_rgb, percentile)
            image_rgb /= np.percentile(image_rgb, 100 - percentile)
        elif brightness == 'shift':
            mult = 0.25 / np.mean(image_rgb)
            image_rgb *= mult
            
        image_rgb = image_rgb.clip(0, 1)
            
        # Gamma correction
        if use_gamma:
            image_rgb = np.power(image_rgb, 1/2.2)

        # Clip invisible pixels
        image_rgb = image_rgb[0:raw.metadata.height, 0:raw.metadata.width, :]

        # Clip & rotate canvas, if needed
        if raw.metadata.orientation == 5:
            image_rgb = np.rot90(image_rgb)
        elif raw.metadata.orientation == 6:
            image_rgb = np.rot90(image_rgb, 3)

    if expand:
        image_rgb = np.expand_dims(image_rgb, axis=0)
        
    return image_rgb


def unpack(filename, stack=True, use_wb=True, expand=False):
    """
    Reads a RAW image and return a raw pixel array (either in native resolution or as a RGGB Bayer stack),
    and basic RAW parameters needed for processing (CFA pattern, sRGB conversion tables, and photo white
    balance multipliers).

    :param filename: input RAW image
    :param stack: set to False to return the standard 1-channel RAW instead of the RGGB stack
    :param use_wb: set to False to disable white balancing based on image meta-data
    """
    
    with Raw(filename) as raw:
        raw.unpack()
        image_raw = np.array(raw.raw_image(), dtype=np.float32)

        # Normalization and calibration
        black = raw.data.contents.color.black
        saturation = raw.data.contents.color.maximum

        image_raw = image_raw.astype(np.float32)
        image_raw -= black
        
        uint14_max = 1
        image_raw *= uint14_max / (saturation - black)
        image_raw = np.clip(image_raw, 0, uint14_max)
            
        cfa_pattern = ''.join([''.join(x) for x in raw.color_filter_array]).upper()
        cam_mul = np.array(raw.data.contents.color.cam_mul, dtype=np.float32)
        cam_mul /= cam_mul[1] # Set the multiplier for G to be 1
        
        # White balancing
        if use_wb:
            if cfa_pattern == 'GBRG':
                image_raw[1::2, 0::2] *= cam_mul[0]
                image_raw[0::2, 1::2] *= cam_mul[2]
            elif cfa_pattern == 'RGGB':
                image_raw[0::2, 0::2] *= cam_mul[0]
                image_raw[1::2, 1::2] *= cam_mul[2]
            elif cfa_pattern == 'BGGR':
                image_raw[1::2, 1::2] *= cam_mul[0]
                image_raw[0::2, 0::2] *= cam_mul[2]        
            
        image_raw = image_raw.clip(0, uint14_max)
        cam2srgb = np.array(raw.data.contents.color.rgb_cam, dtype=np.float).reshape((3,4))[:, 0:3]

        if stack:
            if cfa_pattern == 'GBRG':
                r = image_raw[1::2, 0::2]
                g1 = image_raw[0::2, 0::2]
                g2 = image_raw[1::2, 1::2]
                b = image_raw[0::2, 1::2]
                
            elif cfa_pattern == 'RGGB':
                r = image_raw[0::2, 0::2]
                g1 = image_raw[0::2, 1::2]
                g2 = image_raw[1::2, 0::2]
                b = image_raw[1::2, 1::2]
                
            elif cfa_pattern == 'BGGR':
                r = image_raw[1::2, 1::2]
                g1 = image_raw[0::2, 1::2]
                g2 = image_raw[1::2, 0::2]
                b = image_raw[0::2, 0::2]
            else:
                raise ValueError('Unsupported CFA pattern: {}'.format(cfa_pattern))

            image_raw = np.dstack([r, g1, g2, b]).clip(0, 1)

        if expand:
            image_raw = np.expand_dims(image_raw, axis=0)

        return image_raw, cfa_pattern, cam2srgb, cam_mul


def process_auto(filename):
    """
    Process a RAW image using libRAW with default settings.
    :param filename: input RAW image
    """
    with Raw(filename=filename) as raw:
        raw.unpack()
        raw.process()
        image = raw.to_buffer()

        image = np.frombuffer(image, dtype=np.uint8)

        if raw.metadata.orientation == 5:
            image = image.reshape((raw.metadata.width, raw.metadata.height, 3))
        else:
            image = image.reshape((raw.metadata.height, raw.metadata.width, 3))
    
        return image


def stack_bayer(image_rgb, cfa_pattern):
    """
    Return a RGGB Bayer stack sampled from a RGB image according to a given CFA configuration.
    :param image_rgb: 3-D numpy array (h, w, 3:rgb)
    :param cfa_pattern: 'GBRG', 'RGGB' or 'BGGR'
    """
    cfa_pattern = cfa_pattern.upper()

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

    else:
        raise ValueError(f"Unsupported CFA pattern: {cfa_pattern}")

    return np.dstack([r, g1, g2, b])


def simulate_bayer(image_rgb, cfa_pattern):
    """
    Simulate a Bayer image from full RGB image.
    :param image_rgb: 3-D numpy array (h, w, 3:rgb) or 4-D image batch
    :param cfa_pattern: 'GBRG', 'RGGB' or 'BGGR'
    """
    image_bayer = np.zeros_like(image_rgb)
    cfa_pattern = cfa_pattern.upper()

    if image_rgb.ndim == 3:

        if cfa_pattern.upper() == 'GBRG':
            image_bayer[1::2, 0::2, 0] = image_rgb[1::2, 0::2, 0]
            image_bayer[0::2, 0::2, 1] = image_rgb[0::2, 0::2, 1]
            image_bayer[1::2, 1::2, 1] = image_rgb[1::2, 1::2, 1]
            image_bayer[0::2, 1::2, 2] = image_rgb[0::2, 1::2, 2]

        elif cfa_pattern.upper() == 'RGGB':
            image_bayer[0::2, 0::2, 0] = image_rgb[0::2, 0::2, 0]
            image_bayer[0::2, 1::2, 1] = image_rgb[0::2, 1::2, 1]
            image_bayer[1::2, 0::2, 1] = image_rgb[1::2, 0::2, 1]
            image_bayer[1::2, 1::2, 2] = image_rgb[1::2, 1::2, 2]

        elif cfa_pattern.upper() == 'BGGR':
            image_bayer[1::2, 1::2, 0] = image_rgb[1::2, 1::2, 0]
            image_bayer[0::2, 1::2, 1] = image_rgb[0::2, 1::2, 1]
            image_bayer[1::2, 0::2, 1] = image_rgb[1::2, 0::2, 1]
            image_bayer[0::2, 0::2, 2] = image_rgb[0::2, 0::2, 2]

        else:
            raise ValueError(f"Unsupported CFA pattern: {cfa_pattern}")

    elif image_rgb.ndim == 4:
        for n in range(len(image_rgb)):
            image_bayer[n] = simulate_bayer(image_rgb[n], cfa_pattern)
    else:
        raise ValueError('Unsupported array shape!')

    return image_bayer


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

    cfa_pattern = cfa_pattern.upper()

    assert bayer_stack.ndim == 3

    h, w = bayer_stack.shape[0:2]

    image_rgb = np.zeros((2*h, 2*w, 3), dtype=bayer_stack.dtype)

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

    else:
        raise ValueError(f"Unsupported CFA pattern: {cfa_pattern}")

    return image_rgb
