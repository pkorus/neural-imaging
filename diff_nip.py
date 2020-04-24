#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse

import numpy as np

from helpers import fsutil, imdiff

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def compare_nips(model_a_dirname, model_b_dirname, camera=None, image=None, patch_size=128, root_dirname='./data', output_dir=None, model_a_args=None, model_b_args=None, extras=False):
    """
    Display a comparison of two variants of a neural imaging pipeline.
    :param camera: camera name (e.g., 'Nikon D90')
    :param model_a_dirname: directory with the first variant of the model
    :param model_b_dirname: directory with the second variant of the model
    :param ps: patch size (patch will be taken from the middle)
    :param image_id: index of the test image
    :param root_dir: root data directory
    :param output_dir: set an output directory if the figure should be saved (matplotlib2tikz will be used)
    """
    # Lazy imports to minimize delay for invalid command line parameters
    import re
    import inspect
    import imageio as io
    import matplotlib.pyplot as plt

    import tensorflow as tf
    from models import pipelines, tfmodel
    from helpers import raw, loading

    supported_cameras = fsutil.listdir(os.path.join(root_dirname, 'models', 'nip'), '.*')
    supported_pipelines = pipelines.supported_models

    if patch_size > 0 and (patch_size < 8 or patch_size > 2048):
        raise ValueError('Patch size seems to be invalid!')

    if camera is not None and camera not in supported_cameras:
        raise ValueError('Camera data not found ({})! Available cameras: {}'.format(camera, ', '.join(supported_cameras)))

    # Check if the image is an integer
    try:
        image = int(image)
    except:
        pass

    # Construct the NIP models
    if os.path.isdir(model_a_dirname):
        # Restore a NIP model from a training log
        model_a = tfmodel.restore(model_a_dirname, pipelines)
    else:
        # Construct the NIP model from class name (and optional arguments)
        if model_a_args is None:
            model_a = getattr(pipelines, model_a_dirname)()
        else:
            model_a = getattr(pipelines, model_a_dirname)(**model_a_args)
        model_a.load_model(os.path.join(root_dirname, model_a.model_code))

    if os.path.isdir(model_b_dirname):
        # Restore a NIP model from a training log
        model_b = tfmodel.restore(model_b_dirname, pipelines)
    else:
        # Construct the NIP model from class name (and optional arguments)
        if model_b_args is None:
            model_b = getattr(pipelines, model_b_dirname)()
        else:
            model_b = getattr(pipelines, model_b_dirname)(**model_b_args)
        model_b.load_model(os.path.join(root_dirname, model_b.model_code))

    print('ISP-A: {}'.format(model_a.summary()))
    print('ISP-B: {}'.format(model_b.summary()))

    # Load sample data

    if isinstance(image, int) and camera is not None:

        data_dirname = os.path.join(root_dirname, 'raw', 'training_data', camera)
        files = fsutil.listdir(data_dirname, '.*\.png')
        files = files[image:image+1]
        print('Loading image {} from the training set: {}'.format(image, files))
        data = loading.load_images(files, data_dirname)
        sample_x, sample_y = data['x'].astype(np.float32) / (2**16 - 1), data['y'].astype(np.float32) / (2**8 - 1)

        with open('config/cameras.json') as f:
            cameras = json.load(f)
            cfa, srgb = cameras[camera]['cfa'], np.array(cameras[camera]['srgb'])

        image = files[0]

    elif image is not None:
        print('Loading a RAW image {}'.format(image))
        sample_x, cfa, srgb, _ = raw.unpack(image, expand=True)
        sample_y = raw.process(image, brightness=None, expand=True)
        image = os.path.split(image)[-1]

    if isinstance(model_a, pipelines.ClassicISP):
        print('Configuring ISP-A to CFA: {} & sRGB {}'.format(cfa, srgb.round(2).tolist()))
        model_a.set_cfa_pattern(cfa)
        model_a.set_srgb_conversion(srgb)

    if isinstance(model_b, pipelines.ClassicISP):
        print('Configuring ISP-B to CFA: {} & sRGB {}'.format(cfa, srgb.round(2).tolist()))
        model_b.set_cfa_pattern(cfa)
        model_b.set_srgb_conversion(srgb)

    # Develop images
    sample_ya = model_a.process(sample_x).numpy()
    sample_yb = model_b.process(sample_x).numpy()

    if patch_size > 0:
        print('Cropping a {p}x{p} patch from the middle'.format(p=patch_size))
        xx = (sample_x.shape[2] - patch_size // 2) // 2
        yy = (sample_x.shape[1] - patch_size // 2) // 2
        sample_x = sample_x[:, yy:yy+patch_size, xx:xx+patch_size, :]
        sample_y = sample_y[:, 2*yy:2*(yy+patch_size), 2*xx:2*(xx+patch_size), :]
        sample_ya = sample_ya[:, 2*yy:2*(yy+patch_size), 2*xx:2*(xx+patch_size), :]
        sample_yb = sample_yb[:, 2*yy:2*(yy+patch_size), 2*xx:2*(xx+patch_size), :]

    # Plot images
    fig = imdiff.compare_ab_ref(sample_y, sample_ya, sample_yb, fig=plt.figure(), extras=extras)

    if output_dir is not None:
        from tikzplotlib import save as tikz_save
        dcomp = [x for x in fsutil.split(model_b_dirname) if re.match('(ln-.*|[0-9]{3})', x)]
        tikz_save('{}/examples_{}_{}_{}_{}.tex'.format(output_dir, camera, image, model_a_dirname, model_b_dirname), figureheight='8cm', figurewidth='8cm', strict=False)
    else:
        fig.tight_layout()
        fig.show(fig)

    fig.suptitle('{}, A={}, B={}'.format(image, model_a.model_code, model_b.model_code))
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('-c', '--cam', dest='camera', action='store', help='camera')
    parser.add_argument('-i', '--image', dest='image', action='store', 
                        help='RAW image path or training image id')
    parser.add_argument('-p', '--patch', dest='patch', action='store', default=128, type=int,
                        help='patch size')
    parser.add_argument('-a', dest='model_a_dir', action='store', default='./data/models/nip',
                        help='path to first model (TF checkpoint dir)')
    parser.add_argument('-b', dest='model_b_dir', action='store', default='./data/models/nip',
                        help='path to second model (TF checkpoint dir)')
    parser.add_argument('--dir', dest='dir', action='store', default='./data/',
                        help='root directory with images and training data')
    parser.add_argument('-e', '--extra', dest='extras', action='store_true', default=False,
                        help='show additional plots (FFTs and diffs)')
    parser.add_argument('--out', dest='out', action='store', default=None,
                        help='output directory for TikZ output (if set, the figure is not displayed)')
    parser.add_argument('--ha', dest='ha', default=None, help='Set hyper-parameters / override CSV settings for model A (JSON string)')
    parser.add_argument('--hb', dest='hb', default=None, help='Set hyper-parameters / override CSV settings for model A (JSON string)')

    args = parser.parse_args()

    try:
        if args.ha is not None: args.ha = json.loads(args.ha.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.ha.replace('\'', '"'))
        sys.exit(2)

    try:
        if args.hb is not None: args.hb = json.loads(args.hb.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.hb.replace('\'', '"'))
        sys.exit(2)        

    compare_nips(args.model_a_dir, args.model_b_dir, args.camera, args.image,
                 args.patch, args.dir, args.out, args.ha, args.hb, extras=args.extras)


if __name__ == "__main__":
    main()

