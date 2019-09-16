#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import sys
import json
import logging
import argparse
import numpy as np
from pathlib import Path

from helpers import coreutils, dataset, results_data
from training.manipulation import construct_models
from training.validation import confusion
from compression import afi

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('test')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

supported_pipelines = ['UNet', 'DNet', 'INet']


def validate_fan(output_directory, manipulations, data, patch=64, dcn_model=None, downsampling='pool', jpeg_quality=50):

    # Define the distribution channel ----------------------------------------------------------------------------------
    compression_params = {}
    if jpeg_quality is not None:
        compression_params['quality'] = jpeg_quality
        compression_params['rounding_approximation'] = 'soft'
    else:
        if dcn_model in afi.dcn_presets:
            dcn_model = afi.dcn_presets[dcn_model]
        compression_params['dirname'] = dcn_model

    if jpeg_quality is not None:
        compression = 'jpeg'
    elif dcn_model is not None:
        compression = 'dcn'
    else:
        compression = 'none'

    # Parse manipulations
    manipulations = manipulations or ['sharpen', 'resample', 'gaussian', 'jpeg']

    distribution_spec = {
        'downsampling': downsampling,
        'compression': compression,
        'compression_params': compression_params
    }

    # Construct the TF model
    tf_ops, distribution = construct_models('ONet', patch_size=patch, trainable=set(), distribution=distribution_spec, manipulations=manipulations, loss_metric='L2')

    # Load pre-trained models
    if 'dirname' in distribution['compression_params']: tf_ops['dcn'].load_model(
        distribution['compression_params']['dirname'])
    tf_ops['fan'].load_model(os.path.join(output_directory))

    # Create a function which generates labels for each batch
    def batch_labels(batch_size, n_classes):
        return np.concatenate([x * np.ones((batch_size,), dtype=np.int32) for x in range(n_classes)])

    n_classes = len(distribution['forensics_classes'])

    # Compute the confusion matrix
    conf_mat = confusion(tf_ops['fan'], data, lambda x: batch_labels(x, n_classes))
    return conf_mat, distribution['forensics_classes']


def main():
    parser = argparse.ArgumentParser(description='Test manipulation detection (FAN) on RGB images')
    group = parser.add_argument_group('General settings')
    group.add_argument('--patch', dest='patch', action='store', default=64, type=int,
                        help='patch size')
    group.add_argument('--patches', dest='patches', action='store', default=1, type=int,
                        help='number of validation patches')
    group.add_argument('--data', dest='data', action='store', default='./data/rgb/32k',
                        help='directory with test RGB images')

    group = parser.add_argument_group('Training session selection')
    group.add_argument('--dir', dest='dir', action='store', default='./data/raw/m',
                        help='directory with training sessions')
    group.add_argument('--re', dest='re', action='store', default=None,
                        help='regular expression to filter training sessions')

    group = parser.add_argument_group('Override training settings')
    group.add_argument('--jpeg', dest='jpeg_quality', action='store', default=None, type=int,
                        help='Override JPEG quality level (distribution channel)')
    group.add_argument('--dcn', dest='dcn_model', action='store', default=None,
                        help='DCN compression model path')
    group.add_argument('--manip', dest='manipulations', action='store', default=None,
                       help='Included manipulations, e.g., : {}'.format('sharpen,jpeg,resample,gaussian'))

    args = parser.parse_args()

    # Split manipulations
    if args.manipulations is not None:
        args.manipulations = args.manipulations.strip().split(',')

    json_files = sorted(str(f) for f in Path(args.dir).glob('**/training.json'))

    if len(json_files) == 0:
        sys.exit(0)

    # Load training / validation data
    data = dataset.IPDataset(args.data, n_images=0, v_images=-1, load='y', val_rgb_patch_size=2 * args.patch, val_n_patches=args.patches)

    print('Found {} candidate training sessions ({})'.format(len(json_files), args.dir))
    print('Data: {}'.format(data.description))

    for filename in json_files:
        if args.re is None or re.findall(args.re, filename):

            with open(filename) as f:
                training_log = json.load(f)

            # Setup manipulations
            if args.manipulations is None:
                manipulations = eval(coreutils.getkey(training_log, 'summary/Classes'))
                # TODO More elegant solution needed
                manipulations.remove('native')
                if 'jpg' in manipulations:
                    manipulations.append('jpeg')
                    manipulations.remove('jpg')
            else:
                manipulations = args.manipulations

            accuracy = coreutils.getkey(training_log, 'forensics/validation/accuracy')[-1]
            compression = coreutils.getkey(training_log, 'summary/Channel Compression')
            downsampling = coreutils.getkey(training_log, 'summary/Channel Downsampling')

            # Setup compression
            if compression.startswith('jpeg'):
                # Override from CLI arguments or read from training log
                if args.jpeg is not None:
                    jpeg = args.jpeg
                else:
                    jpeg = int(re.findall('\(([0-9]+),', compression)[0])
                dcn_model = None
            else:
                jpeg = None

                if args.dcn_model is not None:
                    # Override from CLI arguments
                    dcn_model = args.dcn_model
                else:
                    # Otherwise, read from the training log
                    if 'dcn' in coreutils.getkey(training_log, 'summary/Joint optimization'):
                        # If the DCN is trainable, load the fine-tuned model
                        dcn_model = os.path.join(os.path.split(filename)[0], 'models')
                    else:
                        # If not trainable, load a baseline DCN model
                        compression_params = eval(coreutils.getkey(training_log, 'summary/Channel Compression Parameters'))
                        dcn_model = compression_params['dirname']

            print('\n> {}'.format(filename))
            print('Compression: {}'.format(compression))
            print('Downsampling: {}'.format(downsampling))

            conf_mat, labels = validate_fan(os.path.join(os.path.split(filename)[0], 'models'), manipulations, data, args.patch, dcn_model, downsampling, jpeg)

            print(results_data.confusion_to_text((100*conf_mat).round(0), labels, filename, 'txt'))
            print('Accuracy: {:.2f} // Expected {:.2f}'.format(np.mean(np.diag(conf_mat)), accuracy))


if __name__ == "__main__":
    main()
