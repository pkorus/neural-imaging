import os
import re
import sys
import shutil
from helpers import coreutils

from pathlib import Path

run = 0
dirname = './data/m/7-rgb/dcn+'
output = './data/dcn/forensics/7-rgb'
basics = './data/dcn/_all/entropy'

dcn_presets = {
    '16c': 'TwitterDCN-4096D/16x16x16-r:soft-codebook-Q-5.0bpf-S+-H+250.00',
    '32c': 'TwitterDCN-8192D/16x16x32-r:soft-codebook-Q-5.0bpf-S+-H+250.00',
    '64c': 'TwitterDCN-16384D/16x16x64-r:soft-codebook-Q-5.0bpf-S+-H+250.00'
}

jsons_files = sorted(str(f) for f in Path(dirname).glob('**/training.json'))

models = set()
destinations = {}

for file in jsons_files:

    # Find the optimized models
    model = re.findall('([0-9]+k)', file)[0]
    models.add(model)
    strength = re.findall('lc-([0-9\.]+)', file)[0]
    tf_model = os.path.join(os.path.split(file)[0], 'models/twitterdcn')
    print(model, strength, tf_model, os.path.exists(tf_model))

    tgt_dirname = os.path.join(output, '{}-{}'.format(model, strength))
    if model not in destinations:
        destinations[model] = []
    destinations[model].append(tgt_dirname)

    # Copy trained models
    if not os.path.isdir(tgt_dirname):
        os.makedirs(tgt_dirname)

    if not os.path.isdir(os.path.join(tgt_dirname, 'twitterdcn')):
        shutil.copytree(tf_model, os.path.join(tgt_dirname, 'twitterdcn'))

# Add basic models

for model in models:
    tf_model = os.path.join(basics, dcn_presets[model], 'twitterdcn')
    tgt_dirname = os.path.join(output, '{}-{}'.format(model, 'basic'))

    # Copy trained models
    if not os.path.isdir(tgt_dirname):
        os.makedirs(tgt_dirname)

    if not os.path.isdir(os.path.join(tgt_dirname, 'twitterdcn')):
        shutil.copytree(tf_model, os.path.join(tgt_dirname, 'twitterdcn'), ignore=shutil.ignore_patterns('*.png'))

    # Copy JSON model configurations
    for dst in destinations[model]:
        shutil.copy(
            os.path.join(tf_model, 'progress.json'),
            os.path.join(dst, 'twitterdcn', 'progress.json')
        )
