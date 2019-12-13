import os
import re
import sys
import shutil
from helpers import coreutils

from pathlib import Path

run = 0
dirname = './data/m/7-raw/dcn+'
output = './data/models/dcn/_forensics_exp/7-raw-all-reps'
basics = './data/models/dcn/baselines'

dcn_presets = {
    '16c': 'TwitterDCN-4096D/16x16x16-r:soft-codebook-Q-5.0bpf-S+-H+250.00',
    '32c': 'TwitterDCN-8192D/16x16x32-r:soft-codebook-Q-5.0bpf-S+-H+250.00',
    '64c': 'TwitterDCN-16384D/16x16x64-r:soft-codebook-Q-5.0bpf-S+-H+250.00'
}

jsons_files = sorted(str(f) for f in Path(dirname).glob('**/training.json'))

models = set()
destinations = {}

last_strength = None
letter_index = -1
letters = 'abcdefghijklmnopqrstuvwxyz'

for file in jsons_files:

    # Find the optimized models
    model = re.findall('([0-9]+c)', file)[0]
    models.add(model)
    strength = re.findall('lc-([0-9\.]+)', file)[0]
    tf_model = os.path.join(os.path.split(file)[0], 'models/twitterdcn')

    assert os.path.exists(tf_model)

    if model not in destinations:
        destinations[model] = []

    if last_strength == strength or last_strength is None:
        letter_index += 1
    else:
        letter_index = 0

    tgt_dirname = os.path.join(output, '{}-{}-{}'.format(model, strength, letters[letter_index]))
    print(model, strength, tf_model, '->', tgt_dirname)

    last_strength = strength
    destinations[model].append(tgt_dirname)

    # Copy trained models
    if not os.path.isdir(tgt_dirname):
        os.makedirs(tgt_dirname)

    if not os.path.isdir(os.path.join(tgt_dirname, 'twitterdcn')):
        shutil.copytree(tf_model, os.path.join(tgt_dirname, 'twitterdcn'))

# Add basic models

for model in models:
    tf_model = os.path.join(basics, model, 'twitterdcn')
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
