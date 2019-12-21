import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import coreutils

ROOT_DIRNAME = './data/m/5-raw/cvpr2019'


def autodetect_cameras(dirname):
    counter = 5
    while counter > 0 and not os.path.exists(os.path.join(dirname, 'models', 'nip')):
        dirname = os.path.split(dirname)[0]
        counter -= 1

    if counter == 0:
        raise ValueError('The {} directory does not seem to be a valid results directory'.format(dirname))

    return coreutils.listdir(os.path.join(dirname, 'models', 'nip'), '.*', dirs_only=True)


def manipulation_metrics(nip_models, cameras, root_dir=ROOT_DIRNAME):

    nip_models = [nip_models] if type(nip_models) is str else nip_models
    cameras = cameras or coreutils.listdir(root_dir, '.', dirs_only=True)

    if any(cam not in autodetect_cameras(root_dir) for cam in cameras):
        raise ValueError('The auto-detected camera list does not seem to make sense: {}'.format(cameras))

    df = pd.DataFrame(columns=['camera', 'nip', 'ln', 'source', 'psnr', 'ssim', 'accuracy'])

    for camera in cameras:

        nip_models = nip_models or coreutils.listdir(os.path.join(root_dir, camera), '.', dirs_only=True)

        for nip in nip_models:

            find_dir = os.path.join(root_dir, camera, nip)
            experiment_dirs = coreutils.listdir(os.path.join(find_dir), '.*', dirs_only=True)

            for ed in experiment_dirs:

                exp_dir = os.path.join(find_dir, ed)
                jsons_files = sorted(str(f) for f in Path(exp_dir).glob('**/training.json'))

                for jf in jsons_files:
                    with open(jf) as f:
                        data = json.load(f)

                    df = df.append({'camera': camera,
                                    'nip': nip,
                                    'ln': ed,
                                    'source': jf.replace(find_dir, '').replace('training.json', ''),
                                    'psnr': data['nip']['validation']['psnr'][-1],
                                    'ssim': data['nip']['validation']['ssim'][-1],
                                    'accuracy': data['forensics']['validation']['accuracy'][-1]
                        }, ignore_index=True)

    return df


def manipulation_progress(cases, root_dir=ROOT_DIRNAME):

    cases = cases or [('Nikon D90', 'INet', 'lr-0.0000', 0)]

    df = pd.DataFrame(columns=['camera', 'nip', 'exp', 'rep', 'step', 'psnr', 'ssim', 'accuracy'])
    labels = []

    l_camera, l_nip_model, l_ed, l_rep = None, None, None, None

    for i, (camera, nip_model, ed, rep) in enumerate(cases):

        # If something is unspecified, use the last known value
        camera = camera or l_camera
        nip_model = nip_model or l_nip_model
        ed = ed or l_ed
        rep = rep if rep is not None else l_rep

        filename = os.path.join(root_dir, camera, nip_model, ed, '{:03d}'.format(rep), 'training.json')

        if not os.path.isfile(filename):
            print('! warning: could not find file {}'.format(filename))
            continue

        labels.append('{0} ({1}/{2}/{3})'.format(camera, nip_model, ed, rep))

        with open(filename) as f:
            data = json.load(f)

        def match_length(y, x):
            x = x[:len(y)]
            for _ in range(len(y) - len(x)):
                x.append(x[-1])
            return x

        d_psnr = data['nip']['validation']['psnr']
        d_ssim = data['nip']['validation']['ssim']
        d_accuracy = data['forensics']['validation']['accuracy']

        df = df.append(pd.DataFrame({
            'camera': [camera] * len(d_accuracy),
            'nip': [nip_model] * len(d_accuracy),
            'exp': [ed] * len(d_accuracy),
            'rep': [rep] * len(d_accuracy),
            'step': list(range(len(d_accuracy))),
            'psnr': match_length(d_accuracy, d_psnr),
            'ssim': match_length(d_accuracy, d_ssim),
            'accuracy': d_accuracy
        }), ignore_index=True, sort=False)

        # Remember last used values for future iterations
        l_camera, l_nip_model, l_ed, l_rep = camera, nip_model, ed, rep

    if len(df) == 0:
        raise RuntimeError('Empty dataframe! Double check experimental scenario!')

    return df, labels


def manipulation_summary(dirname):
    df = pd.DataFrame(columns=['scenario', 'run', 'accuracy', 'nip_ssim', 'nip_psnr', 'dcn_ssim', 'dcn_entropy'])
    for filename in Path(dirname).glob('**/training.json'):
        with open(str(filename)) as f:
            data = json.load(f)

        default = [np.nan]
        accuracy = coreutils.getkey(data, 'forensics/validation/accuracy') or default
        nip_ssim = coreutils.getkey(data, 'nip/validation/ssim') or default
        nip_psnr = coreutils.getkey(data, 'nip/validation/psnr') or default
        dcn_ssim = coreutils.getkey(data, 'compression/validation/ssim') or default
        dcn_entr = coreutils.getkey(data, 'compression/validation/entropy') or default

        path_components = coreutils.splitall(os.path.relpath(str(filename), dirname))[:-1]

        df = df.append({
            'scenario': os.path.join(*path_components[:-1]),
            'run': int(path_components[-1]),
            'accuracy': accuracy[-1],
            'nip_ssim': nip_ssim[-1],
            'nip_psnr': nip_psnr[-1],
            'dcn_ssim': dcn_ssim[-1],
            'dcn_entropy': dcn_entr[-1]
        }, ignore_index=True, sort=False)
    return df


coreutils.logCall
def confusion_data(run=None, root_dir=ROOT_DIRNAME):

    confusion = OrderedDict()

    jsons_files = sorted(str(f) for f in Path(root_dir).glob('**/training.json'))

    # Pre-filter only some run numbers
    if run is None:
        print('INFO Using the first found repetition of the experiment')
        run = 0

    jsons_files = [jf for jf in jsons_files if '/{:03d}/'.format(run) in jf]

    for jf in jsons_files:

        with open(jf) as f:
            data = json.load(f)

        confusion['{}'.format(os.path.relpath(os.path.split(jf)[0], root_dir)).replace('/{:03d}'.format(run), '')] = {
            'data': np.array(data['forensics']['validation']['confusion']),
            'labels': data['summary']['Classes'] if isinstance(data['summary']['Classes'], list) else eval(data['summary']['Classes'])
        }

    return confusion


def confusion_to_text(conf, labels, title='', fmt='txt'):
    if not isinstance(conf, np.ndarray):
        conf = np.array(conf)

    if conf.ndim != 2:
        raise ValueError('2D array expected!')

    n = conf.shape[0]
    l = max([len(x) for x in labels])

    # Append the pre-amble
    out = []

    if fmt == 'tex':
        out.append('\\documentclass[preview]{standalone}\n')
        out.append('\\usepackage{booktabs}\n')
        out.append('\\usepackage{diagbox}\n')
        out.append('\\usepackage{graphicx}\n')
        out.append('\\usepackage{xcolor,colortbl}\n')
        out.append('\\begin{document}\n')
        out.append('\\begin{preview}\n')
        out.append('\\begin{{tabular}}{{l{0}}}\n'.format(n * 'r'))
        out.append('\\multicolumn{{{0}}}{{c}}{{{1} $\\rightarrow$ {2}\\%}} '.format(n + 1, title, np.mean(np.diag(conf))))
        out.append('\\tabularnewline\n')
        out.append('\\diagbox{\\textbf{True}}{\\textbf{Predicted}}')

        # Fill the header with class names
        for i in range(n):
            out.append('& \\rotatebox{{90}}{{\\textbf{{{0}}}}}'.format(labels[i]))
        out.append(' \\tabularnewline\n')
        out.append('\\toprule\n')

        for i in range(n):
            out.append('\\textbf{{{0}}} '.format(labels[i]))
            for j in range(n):
                if conf[i][j] == 0:
                    out.append('& ')
                elif conf[i][j] < 3:
                    out.append('& *')
                else:
                    out.append('& \\cellcolor{{{0}!{1:.0f}}} {1:.0f}'.format('lime' if i == j else 'red', conf[i][j]))
            out.append(' \\tabularnewline\n')

        out.append('\\bottomrule\n')
        out.append('\\end{tabular}\n')
        out.append('\\end{preview}\n')
        out.append('\\end{document}\n')

    elif fmt == 'txt':
        out.append('# {} (acc={:.1f})'.format(title, np.mean(np.diag(conf))))
        out.append('\n')
        out.append(' ' * l)
        for i in range(n):
            out.append('{:>4}'.format(labels[i][0]))
        out.append('\n')
        for i in range(n):
            out.append('{:>{width}}'.format(labels[i], width=l))
            for j in range(n):
                out.append('{:4.0f}'.format(conf[i][j]))
            out.append('\n')

    else:
        raise ValueError('Invalid format! Only `tex` and `txt` are supported.')

    return ''.join(out)


def nip_stats(dirname, n=1):

    cameras = sorted(os.listdir(dirname))
    df = pd.DataFrame(columns=['pipeline', 'camera', 'psnr', 'ssim'])

    for camera in cameras:
        pipelines = sorted(os.listdir(os.path.join(dirname, camera)))

        for pipe in pipelines:
            with open(os.path.join(dirname, camera, pipe, 'progress.json')) as f:
                ts = json.load(f)

            data = ts if 'psnr' in ts else ts['Performance']

            df = df.append({
                'pipeline': pipe,
                'camera': camera,
                'psnr': np.mean(np.mean(data['psnr'][-n:])),
                'ssim': np.mean(np.mean(data['ssim'][-n:]))
            }, ignore_index=True, sort=False)

    return df