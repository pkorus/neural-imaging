#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import argparse
import numpy as np

from helpers import fsutil, dataset, metrics, plots, raw
from models import tfmodel, pipelines

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def develop_image(pipeline, camera=None, batch=None, image=None, patch_size=0, patches=2, root_dir='./data', pipeline_args=None):
    """
    Display a patch developed by a neural imaging pipeline.
    """

    if camera is not None:
        supported_cameras = fsutil.listdir(os.path.join(root_dir, 'models', 'nip'), '.*')
        if camera not in supported_cameras:
            raise ValueError('Camera data not found ({})! Available cameras: {}'.format(camera, ', '.join(supported_cameras)))
        root_dirname = os.path.join(root_dir, 'models', 'nip', camera)
        data_dirname = os.path.join(root_dir, 'raw', 'training_data', camera)

    if patch_size != 0 and (patch_size < 4 or patch_size > 2048):
        raise ValueError('Patch size seems to be invalid!')

    # Lazy imports to minimize delay for invalid command line parameters
    import numpy as np
    import imageio as io
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from models import pipelines

    # Construct the NIP model ---------------------------------------------------------------------
    
    if os.path.isdir(pipeline):
        # Restore a NIP model from a training log
        model = tfmodel.restore(pipeline, pipelines)
    else:
        # Construct the NIP model from class name (and optional arguments)
        if pipeline_args is None:
            model = getattr(pipelines, pipeline)()
        else:
            model = getattr(pipelines, pipeline)(**pipeline_args)

        loaded_model = False
        candidate_dirs = [os.path.join(root_dirname, model.model_code), os.path.join(root_dirname)]
        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                model.load_model(candidate)
                loaded_model = True
                break

        if not loaded_model:
            raise FileNotFoundError(f'Could not find the corresponding model: {candidate_dirs}')

    # Load image(s) -------------------------------------------------------------------------------
    
    if image is None and batch is not None:
        print('Loading a batch of {} images'.format(batch))
        data = dataset.Dataset(data_dirname, n_images=0, v_images=batch, val_rgb_patch_size=patch_size or 256, val_n_patches=patches)
        sample_x, sample_y = data.next_validation_batch(0, data.count_validation)

        with open('config/cameras.json') as f:
            cameras = json.load(f)
            cfa, srgb = cameras[camera]['cfa'], np.array(cameras[camera]['srgb'])

    elif image is not None:
        print('Loading a RAW image {}'.format(image))
        sample_x, cfa, srgb, _ = raw.unpack(image, expand=True)
        sample_y = raw.process(image, brightness=None, expand=True)

    if isinstance(model, pipelines.ClassicISP):
        print('Configuring ISP to CFA: {} & sRGB {}'.format(cfa, srgb.round(2).tolist()))
        model.set_cfa_pattern(cfa)
        model.set_srgb_conversion(srgb)

    sample_Y = model.process(sample_x).numpy()

    if patch_size > 0:
        xx = (sample_y.shape[2] - patch_size) // 2
        yy = (sample_y.shape[1] - patch_size) // 2
        sample_y = sample_y[:, yy:yy+patch_size, xx:xx+patch_size, :]
        sample_Y = sample_Y[:, yy:yy+patch_size, xx:xx+patch_size, :]

    psnrs = metrics.psnr(sample_y, sample_Y)
    ssims = metrics.ssim(sample_y, sample_Y)

    print('sample x: {}'.format(sample_x.shape))
    print('sample y: {}'.format(sample_y.shape))
    print('sample Y: {}'.format(sample_Y.shape))

    # Plot images ---------------------------------------------------------------------------------
    if len(sample_y) > 1:
        sample_y = plots.thumbnails(sample_y, batch, True)
        sample_Y = plots.thumbnails(sample_Y, batch, True)
    else:
        sample_y = sample_y.squeeze()
        sample_Y = sample_Y.squeeze()

    print('thumbnails: {}'.format(sample_y.shape))

    ncols = 1 if sample_y.shape[1] > sample_y.shape[0] else 2
    nrows = 2 if ncols == 1 else 1
    fig, axes = plt.subplots(nrows, ncols)

    plots.image(sample_Y, '{}, PSNR={:.1f} dB, SSIM={:.2f} : {{}}'.format(model.model_code, float(psnrs.mean()), float(ssims.mean())), axes=axes[0])
    plots.image(sample_y, 'Target RGB images () : {}', axes=axes[1])

    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Develops RAW images with a selected pipeline')
    parser.add_argument('-n', '--nip', dest='nip', action='store', help='model name / path to a trained ISP model')
    parser.add_argument('-i', '--image', dest='image', action='store', help='path to a RAW image')
    parser.add_argument('-c', '--cam', dest='camera', action='store', help='camera')
    parser.add_argument('-b', '--batch', dest='batch', action='store', default=8, type=int,
                        help='load a batch of images (batch size)')
    parser.add_argument('-t', '--patches', dest='patches', action='store', default=3, type=int,
                        help='number of patches per image')
    parser.add_argument('-p', '--patch', dest='patch', action='store', default=0, type=int,
                        help='patch size')
    parser.add_argument('-r', '--dir', dest='dir', action='store', default='./data',
                        help='root directory with images and training data')
    parser.add_argument('--ha', dest='hyperparams_args', default=None, help='Set hyper-parameters / override CSV settings if needed (JSON string)')

    args = parser.parse_args()

    if not args.nip:
        print('Camera ISP not specified!')
        parser.print_usage()
        sys.exit(1)

    try:
        if args.hyperparams_args is not None:
            args.hyperparams_args = json.loads(args.hyperparams_args.replace('\'', '"'))
    except json.decoder.JSONDecodeError:
        print('WARNING', 'JSON parsing error for: ', args.hyperparams_args.replace('\'', '"'))
        sys.exit(2)

    develop_image(args.nip, args.camera, args.batch, args.image, args.patch, args.patches, args.dir, args.hyperparams_args)


if __name__ == "__main__":
    main()
