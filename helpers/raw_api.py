import colour_demosaicing
from rawkit.raw import Raw
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('raw_api')


def stacked_bayer(filename, use_wb=True):
    """
    Get a RGGB Bayer stack from a RAW image.
    :param filename: RAW image
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

        # CFA sanity check
        cfa_pattern = ''.join([''.join(x) for x in raw.color_filter_array])
        
        if cfa_pattern not in ['GBRG', 'RGGB', 'BGGR']:
            raise ValueError('Unsupported CFA configuration: {}'.format(cfa_pattern))

        # White balancing
        if use_wb:
            cam_mul = np.array(raw.data.contents.color.cam_mul, dtype=np.float32)
            cam_mul /= cam_mul[1] # Enfore the multiplier for G to be 1
                    
            if cfa_pattern == 'GBRG':    
                image_raw[1::2, 0::2] *= cam_mul[0]
                image_raw[0::2, 1::2] *= cam_mul[2]
                
            elif cfa_pattern == 'RGGB':    
                image_raw[0::2, 0::2] *= cam_mul[0]
                image_raw[1::2, 1::2] *= cam_mul[2]
                            
            elif cfa_pattern == 'BGGR':    
                image_raw[1::2, 1::2] *= cam_mul[0]
                image_raw[0::2, 0::2] *= cam_mul[2]
                
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

    return np.dstack([r, g1, g2, b]).clip(0, 1)


def process(filename, use_srgb=True, use_gamma=True, brightness='percentile', demosaicing='menon'):
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
        
        log.debug('Model : {} {}'.format(raw.metadata.make.decode(), raw.metadata.model.decode()))
        log.debug('CFA   : {}'.format(raw.color_description.decode()))    
        
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
        
    return image_rgb


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
