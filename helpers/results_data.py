# -*- coding: utf-8 -*-
""" 
Helper functions & classes to work with results.

Useful functions to display data:
--------------------------------
- confusion_to_text - renders a confusion matrix (+labels) as txt or tex
- convert_table     - renders a 2d array as txt, tex, csv or pd.dataframe
- render_tex        - renders a LaTeX snipped as file / bytes / bitmap / matplotlib figure

Working with results:
---------------------
- load              - load results from JSON / NPZ
- save              - save dict-like results in JSON / NPZ
- ResultCache       - helper class to store and access saved results (uses a filename formatting convention)
"""

import json
import os
import imageio
from collections import OrderedDict
from pathlib import Path
from string import Formatter
from loguru import  logger

import numpy as np
import pandas as pd

from helpers import fsutil, utils

ROOT_DIRNAME = './data/m/5-raw/cvpr2019'


class DefaultFormatter(Formatter):

    def __init__(self, default=None):
        self.default = default

    def get_value(self, key, args, kwds):

        if isinstance(key, str):
            try:
                return kwds[key]
            except KeyError:
                return f'{{{key}}}' if self.default is None else self.default
        else:
            return Formatter.get_value(key, args, kwds)


def autodetect_cameras(dirname):
    """ Returns a list of known cameras (based on available NIP). """
    
    counter = 5
    while counter > 0 and not os.path.exists(os.path.join(dirname, 'models', 'nip')):
        dirname = os.path.split(dirname)[0]
        counter -= 1

    if counter == 0:
        raise ValueError('The {} directory does not seem to be a valid results directory'.format(dirname))

    return fsutil.listdir(os.path.join(dirname, 'models', 'nip'), '.*', dirs_only=True)


def nip_stats(dirname, avg_last_n_runs=1):
    """
    Returns a dataframe with NIP training summary.
    """

    cameras = sorted(os.listdir(dirname))
    df = pd.DataFrame(columns=['pipeline', 'camera', 'psnr', 'ssim'])

    for camera in cameras:
        pipelines = sorted(os.listdir(os.path.join(dirname, camera)))

        for pipe in pipelines:
            with open(os.path.join(dirname, camera, pipe, 'progress.json')) as f:
                ts = json.load(f)

            data = ts if 'psnr' in ts else ts['performance']

            df = df.append({
                'pipeline': pipe,
                'camera': camera,
                'psnr': np.mean(utils.get(data, 'psnr.validation')[-avg_last_n_runs:]),
                'ssim': np.mean(utils.get(data, 'ssim.validation')[-avg_last_n_runs:])
            }, ignore_index=True, sort=False)

    return df


def manipulation_metrics(nip_models, cameras, root_dir=ROOT_DIRNAME):
    """ 
    Returns a dataframe with aggregated metrics from manipulation classification (NIP-specific). 
    """

    nip_models = [nip_models] if type(nip_models) is str else nip_models
    cameras = cameras or fsutil.listdir(root_dir, '.', dirs_only=True)

    if any(cam not in autodetect_cameras(root_dir) for cam in cameras):
        raise ValueError('The list of cameras does not match the auto-detected list of available models: {}'.format(cameras))

    df = pd.DataFrame(columns=['camera', 'nip', 'ln', 'source', 'psnr', 'ssim', 'accuracy'])

    for camera in cameras:

        nip_models = nip_models or fsutil.listdir(os.path.join(root_dir, camera), '.', dirs_only=True)

        for nip in nip_models:

            find_dir = os.path.join(root_dir, camera, nip)
            experiment_dirs = fsutil.listdir(os.path.join(find_dir), '.*', dirs_only=True)

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
                                    'psnr': utils.get(data, 'nip.performance.psnr.validation')[-1],
                                    'ssim': utils.get(data, 'nip.performance.ssim.validation')[-1],
                                    'accuracy': utils.get(data, 'forensics.performance.accuracy.validation')[-1]
                        }, ignore_index=True)

    return df


def manipulation_progress(cases, root_dir=ROOT_DIRNAME):
    """
    Returns a dataframe with summarized classification training progress.
    """

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
            logger.warning(f'Could not find file {filename}')
            continue

        labels.append('{0} ({1}/{2}/{3})'.format(camera, nip_model, ed, rep))

        with open(filename) as f:
            data = json.load(f)

        def match_length(y, x):
            if len(x) == 0:
                x = [np.nan]
            x = x[:len(y)]
            for _ in range(len(y) - len(x)):
                x.append(x[-1])
            return x

        d_psnr = utils.get(data, 'nip.performance.psnr.validation')
        d_ssim = utils.get(data, 'nip.performance.ssim.validation')
        d_accuracy = utils.get(data, 'forensics.performance.accuracy.validation')

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
    """
    Returns a dataframe with aggregated metrics from manipulation classification (generic). 
    """
    df = pd.DataFrame(columns=['scenario', 'run', 'accuracy', 'nip_ssim', 'nip_psnr', 'dcn_ssim', 'dcn_entropy'])
    for filename in Path(dirname).glob('**/training.json'):
        with open(str(filename)) as f:
            data = json.load(f)

        default = [np.nan]
        accuracy = utils.get(data, 'forensics.validation.accuracy') or default
        nip_ssim = utils.get(data, 'nip.validation.ssim') or default
        nip_psnr = utils.get(data, 'nip.validation.psnr') or default
        dcn_ssim = utils.get(data, 'compression.validation.ssim') or default
        dcn_entr = utils.get(data, 'compression.validation.entropy') or default

        path_components = fsutil.split(os.path.relpath(str(filename), dirname))[:-1]

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


def confusion_data(run=None, root_dir=ROOT_DIRNAME):
    """
    Returns a dictionary of all confusion matrices found under a given directory (recursive):
    
    '{normalized-directory-path}' : {
        'data': N x N confusion matrix,
        'labels': names of the classes,
    }

    Note: assumes the directory structure has a 3-digit run number, e.g, /000/ in the path: 
    """

    confusion = OrderedDict()

    jsons_files = sorted(str(f) for f in Path(root_dir).glob('**/training.json'))

    # Pre-filter only some run numbers
    if run is None:
        logger.info('Using the first found repetition of the experiment')
        run = 0

    jsons_files = [jf for jf in jsons_files if '/{:03d}/'.format(run) in jf]

    for jf in jsons_files:

        with open(jf) as f:
            data = json.load(f)

        confusion['{}'.format(os.path.relpath(os.path.split(jf)[0], root_dir)).replace('/{:03d}'.format(run), '')] = {
            'data': np.array(utils.get(data, 'forensics.performance.confusion')),
            'labels': data['summary']['Classes'] if isinstance(data['summary']['Classes'], list) else eval(data['summary']['Classes'])
        }

    return confusion


def confusion_to_text(conf, labels, title='accuracy', fmt='txt'):
    """
    Converts a confusion matrix and class labels into a human-readable format (txt or tex).
    """
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
        out.append('\\multicolumn{{{0}}}{{c}}{{{1} $\\rightarrow$ {2:.1f}\\%}} '.format(n + 1, title, np.mean(np.diag(conf))))
        out.append('\\tabularnewline\n')
        out.append('\\diagbox{\\textbf{True}}{\\textbf{Predicted}}')

        # Fill the header with class names
        for i in range(n):
            out.append('& \\rotatebox{{90}}{{\\textbf{{{0}}}}}'.format(labels[i][:3]))
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


def convert_table(conf, labels, dim_labels='c\\r', title=None, fmt='txt', dec=0, color1='cyan', color0='white', labels_rows=None):
    """
    Converts a 2D array into a human-readable format (txt, tex, csv or dataframe [df]).
    """
    if not isinstance(conf, np.ndarray):
        conf = np.array(conf)

    if conf.ndim != 2:
        raise ValueError('2D array expected!')

    if '\\' not in dim_labels:
        raise ValueError('Invalid label for array dimensions - need: a \\ b')

    n, m = conf.shape
    l = max([len(x)+2+dec for x in labels + [dim_labels]])

    # If not provided, use the same labels for rows as for columns
    labels_rows = labels_rows or labels

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
        out.append('\\begin{{tabular}}{{l{0}}}\n'.format(m * 'r'))
        if title is not None: 
            out.append('\\multicolumn{{{0}}}{{c}}{{{1}}} '.format(m + 1, title))
            out.append('\\tabularnewline\n')
            out.append('\\toprule\n')
            # out.append('\\midrule\n')/
        else:
            out.append('\\toprule\n')
        out.append('\\diagbox{{\\textbf{{{0}}}}}{{\\textbf{{{1}}}}}'.format(*dim_labels.split('\\')))

        # Fill the header with class names
        for i in range(m):
            out.append('& \\rotatebox{{90}}{{\\textbf{{{0}}}}}'.format(labels[i]))
        out.append(' \\tabularnewline\n')
        out.append('\\toprule\n')

        for i in range(n):
            out.append('\\textbf{{{0}}}'.format(labels_rows[i]))
            for j in range(m):
                if conf[i][j] == 0:
                    out.append(' & ')
                elif conf[i][j] < 3:
                    out.append(' & *')
                else:
                    if color1 is not None and color0 is not None:
                        out.append(' & \\cellcolor{{{0}!{1:.0f}!{2}}} {1:.{dec}f}'.format(color1, conf[i][j], color0, dec=dec))
                    else:
                        out.append(' & {0:.{dec}f}'.format(conf[i][j], dec=dec))
            out.append(' \\tabularnewline\n')

        out.append('\\bottomrule\n')
        out.append('\\end{tabular}\n')
        out.append('\\end{preview}\n')
        out.append('\\end{document}\n')

    elif fmt == 'txt':
        out.append('\n')
        if title is not None: 
            out.append('#{}\n'.format(title))
        out.append('{:>{width}}'.format(dim_labels, width=l))
        for i in range(m):
            out.append('{:>{width}}'.format(labels[i], width=l))
        out.append('\n')
        for i in range(n):
            out.append('{:>{width}}'.format(labels_rows[i], width=l))
            for j in range(m):
                out.append('{:{width}.{dec}f}'.format(conf[i][j], width=l, dec=dec))
            out.append('\n')

    elif fmt == 'csv':
        l = 0
        out.append('\n')
        out.append('{:>{width}}'.format(dim_labels, width=l))
        for i in range(m):
            out.append(',{:>{width}}'.format(labels[i], width=l))
        out.append('\n')
        for i in range(n):
            out.append('{:>{width}}'.format(labels_rows[i], width=l))
            for j in range(m):
                out.append(',{:{width}.{dec}f}'.format(conf[i][j], width=l, dec=dec))
            out.append('\n')

    elif fmt == 'df':
        import pandas as pd
        df = pd.DataFrame(data=conf.round(dec), columns=labels, index=labels_rows[0:n])
        return df

    else:
        raise ValueError('Unknown format: {}'.format(fmt))

    return ''.join(out)


def render_tex(latex, format='fig', filename=None):
    """
    Renders a LaTeX snippet for display in a Jupyter notebook. 

    Output format:
    - file  - saves the rendered document as PDF / PNG (depending on the extension);
              if filename not provided, a random one will be generated.
    - bytes - returns bytes of the rendered PDF
    - array - returns a bitmap of a rendered PDF as numpy array
    - fig   - returns a matplotlib figure with displayed bitmap
    """
    from latex import build_pdf

    if 'documentclass' not in latex:
        latex = r"""
        \documentclass[preview]{standalone}
        \usepackage{booktabs}
        \usepackage{diagbox}
        \usepackage{graphicx}
        \usepackage{xcolor,colortbl}
        \begin{document}
        \begin{preview}
        []
        \end{preview}
        \end{document}
        """.replace('[]', latex)

    pdf = build_pdf(latex)
    
    if format == 'file':
        filename = filename or '/tmp/{}.pdf'.format(''.join(np.random.choice(list('abcdef'), 10, replace=True)))
        
        if filename.endswith('.pdf'):
            with open(filename, 'wb') as f:
                f.write(pdf.data)
            
        elif filename.endswith('.png'):
            from pdf2image import convert_from_bytes
            image = convert_from_bytes(pdf.data)
            imageio.imwrite(filename, image)
        
        return filename
    
    elif format == 'bytes':
        return pdf
    
    elif format == 'array':
        from pdf2image import convert_from_bytes
        return np.array(convert_from_bytes(pdf.data)[0])
    
    elif format == 'fig':
        from pdf2image import convert_from_bytes
        from matplotlib.figure import Figure
        dpi, scale = 300, 0.75
        image = np.array(convert_from_bytes(pdf.data, dpi=dpi)[0])
        fig = Figure(figsize=(scale * image.shape[1] / dpi, scale * image.shape[0] / dpi), dpi=dpi)
        fig.gca().imshow(image)
        fig.gca().set_xticks([])
        fig.gca().set_yticks([])
        fig.gca().axis('off')
        return fig

    else:
        raise ValueError('Unsupported format: {}'.format(format))


def save(results, *, filename=None, prefix=None):
    """ Helper function to save dict-like results in either JSON or NPZ (zipped numpy objects) """

    if filename is None:
        filename = results['filename']

    if prefix is not None:
        filename = os.path.join(prefix, filename)

    os.makedirs(os.path.split(filename)[0], exist_ok=True)
    extension = os.path.splitext(filename)[-1].lower()

    if extension == '.npz':
        np.savez(filename, **results)
    
    elif extension == '.json':
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    else:
        raise ValueError(f'Unsupported format: {extension}')


def load(filename, prefix=None):
    """ Helper function to load results from JSON or NPZ (zipped numpy objects) """

    if prefix is not None:
        filename = os.path.join(prefix, filename)

    extension = os.path.splitext(filename)[-1].lower()

    if extension == '.npz':
        data = np.load(filename, allow_pickle=True)
        return {k: data[k] if data[k].ndim > 0 else data[k].item() for k in data.keys()}
    elif extension == '.json':
        with open(filename) as f:
            return json.load(f)
    else:
        raise ValueError(f'Unsupported format: {extension}')


class ResultCache(object):
    """
    Helper class to facilitate saving/loading/finding results. Uses filename patterns. Supports '*' wildcards.

    Uses filename generation patterns (config/result_patterns.json):
        ['dirname with {arg_1}', 'dirname with {arg_i}', ..., 'filename with {arg_n-1} and {arg_n}']
    e.g.:
        ["baseline_{engine}", "{isp_summary}", "{patch_size}px", "qf_{jpeg_qf}", "{lab_samples}", "results.npz"]

    TLDR usage:

    # Init (from pre-defined filename patterns)
    cache = ResultCache('prnu_sim-detection', 'data/f', patch_size=64, engine='mle')

    # Init (ad hoc)
    cache = ResultCache(["baseline_{engine}", "{patch_size}px", "qf_{jpeg_qf}", "results.npz"], 'data/f', patch_size=64, engine='mle')

    # List all files that match:
    cache.find()

    # Saving results
    cache.save(results, jpeg_qf=90)

    # Loading results
    cache.load(jpeg_qf=90)
    """
    
    def __init__(self, pattern, prefix, **kwargs):
        """
        :param pattern: a string (key to dict in config/result_patterns.json) or iterable (with filename pattern definition)
        :param prefix: file path prefix (e.g., root directory where results are stored)
        :param kwargs: keyword args to narrow down search results (more keywords can be supplied in query functions)

        """
        from collections import Iterable
        self.prefix = prefix
        self._pattern = pattern
        if isinstance(pattern, str):
            with open('config/result_patterns.json') as f:
                result_patterns = json.load(f)
            self.pattern = result_patterns[pattern]
        elif isinstance(pattern, Iterable):
            self.pattern = tuple(pattern)
        self.kwargs = kwargs
        
    def set(self, **kwargs):
        self.kwargs.update(kwargs)

    def unset(self, fields):
        if isinstance(fields, str):
            del self.kwargs[fields]
        else:
            for f in fields:
                del self.kwargs[f]

    def filename(self, **kwargs):
        """ Generate a unique filename for the current context. Raises exception if not unique. Add keyword args to narrow down. """ 
        args = {**self.kwargs}
        args.update(kwargs)
        try:
            filename = os.path.join(self.prefix, *[x.format(**args) for x in self.pattern])
            if '*' in filename:
                raise ValueError('Wildcards found - not a valid filename!')
            return filename
        except:
            pattern = self._get_wildcard_pattern(args)
            candidates = list(str(x) for x in Path('.').glob(pattern))
            if len(candidates) == 1:
                return candidates[0]
            else:
                raise ValueError(f'Current search pattern [{pattern}] must match 1 file but matches {len(candidates)}')

    def load_all(self, **kwargs):
        """ Load all results matching the current search pattern and return a dict indexed by representative filename sections """
        results = OrderedDict()
        filenames = self.find(**kwargs)
        labels = fsutil.strip_prefix(filenames)
        for l, f in zip(labels, filenames):
            results[l] = load(f)
        return results

    def load(self, **kwargs):
        """ Load results for a given context (use extra keyword args to narrow down) """ 
        filename = self.filename(**kwargs)
        return load(filename)
    
    def save(self, results, overwrite=False, **kwargs):
        """ Save results for a given context (use extra keyword args to narrow down) """ 
        filename = self.filename(**kwargs)
        if not overwrite and os.path.isfile(filename):
            raise FileExistsError(f'File {filename} exists! Use overwrite=True if needed.')
        save(results, filename=filename)

    @staticmethod
    def format(pattern, prefix=None, **kwargs):
        if isinstance(pattern, str):
            with open('config/result_patterns.json') as f:
                result_patterns = json.load(f)
            pattern = result_patterns[pattern]
        if prefix is not None:
            return os.path.join(prefix, *[x.format(**kwargs) for x in pattern])
        else:
            return os.path.join(*[x.format(**kwargs) for x in pattern])

    def _get_wildcard_pattern(self, args=None):
        """ Generate a wildcard pattern for the given context """
        fmt = DefaultFormatter('*')
        return os.path.join(self.prefix, *[fmt.format(x, **args) for x in self.pattern])

    def find(self, **kwargs):
        """ Find all files matching the current context """
        args = {**self.kwargs}
        args.update(kwargs)
        fmt = DefaultFormatter('*')
        pattern = os.path.join(self.prefix, *[fmt.format(x, **args) for x in self.pattern])
        logger.info(f'*> {pattern}')
        return list(str(x) for x in Path('.').glob(pattern))

    def __str__(self):
        fmt = DefaultFormatter()
        return '{} <- {}'.format(
            self.__class__.__name__,
            os.path.join(self.prefix, *[fmt.format(x, **self.kwargs) for x in self.pattern])
            )
        
    def __repr__(self):
        return '{}("{}","{}",{})'.format(
            self.__class__.__name__,
            self._pattern,
            self.prefix,
            utils.join_args(self.kwargs)
        )

