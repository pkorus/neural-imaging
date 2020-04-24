# -*- coding: utf-8 -*-
"""
High-level plotting and visualization functions.

Caution
-------
The functions create unmanaged matplotlib figures to prevent RAM leaks. If managed figures are required, they need
to be passed from outside, e.g.,:

plots.images(batch, ..., fig=plt.figure())

Overview
--------
- image             - show a single image (axes-level)
- images            - show a batch of images (figure-level)
- sub               - generate a figure divided into sub-plots (ALWAYS returns a plain list of axes)
- progress          - plot training progress with moving averages (single metric)
- perf              - plot training progress with moving averages (various metrics)
- detection         - binary detection metrics (positive/negative/reference stats)
- roc               - ROC curve with basic stats (tpr, auc)
- intervals         - plots mean values with shaded percentile ranges
- correlation       - scatter plot with correlation stats
- scatter_hex       - 2d density plot with hex binning

"""
import imageio
import numpy as np

from matplotlib.figure import Figure
from skimage.transform import resize

from loguru import logger

from helpers import stats, utils


def configure(profile=None):

    if profile == 'tex':
        from matplotlib import rc
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        rc('axes', titlesize=14)
        rc('axes', labelsize=14)
        rc('xtick', labelsize=8)
        rc('ytick', labelsize=8)
        rc('legend', fontsize=10)
        rc('figure', titlesize=14)
    if profile == "big":
        from matplotlib import rc
        # rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=False)
        rc('axes', titlesize=14)
        rc('axes', labelsize=14)
        rc('xtick', labelsize=12)
        rc('ytick', labelsize=12)
        rc('legend', fontsize=12)
        rc('figure', titlesize=14)
    else:
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)
    # import seaborn as sns
    # sns.set('paper', font_scale=2, style="darkgrid")
    # sns.set_context("paper")


def thumbnails(images, ncols=0, columnwise=False):
    """
    Return a numpy array with image thumbnails.
    """
    
    if type(images) is np.ndarray:
        
        n_images = images.shape[0]        
        n_channels = images.shape[-1]
        img_size = images.shape[1:]
        
        if len(img_size) == 2:
            img_size.append(1)
                        
    elif type(images) is list or type(images) is tuple:
        
        n_images = len(images)
        n_channels = images[0].shape[-1]
        img_size = list(images[0].shape)
        
        if len(img_size) == 2:
            img_size.append(1)
    
    ncols = ncols if ncols > 0 else n_images
    images_x = ncols or int(np.ceil(np.sqrt(n_images)))
    images_y = int(np.ceil(n_images / images_x))
    size = (images_y, images_x)
        
    # Allocate space for the thumbnails
    output = np.zeros((size[0] * img_size[0], size[1] * img_size[1], img_size[2]))
        
    for r in range(n_images):
        bx = int(r % images_x)
        by = int(np.floor(r / images_x))
        if columnwise:
            by = int(r % images_y)
            bx = int(np.floor(r / images_y))
        current = images[r].squeeze()
        if current.shape[0] != img_size[0] or current.shape[1] != img_size[1]:
            current = resize(current, img_size[:-1], anti_aliasing=True)
        if len(current.shape) == 2:
            current = np.expand_dims(current, axis=2)
        output[by*img_size[0]:(by+1)*img_size[0], bx*img_size[1]:(bx+1)*img_size[1], :] = current
        
    return output
    

def _imarray(img, n_images, fetch_hook, titles, figwidth=4, cmap='gray', ncols=0, fig=None, rowlabels=None):
    """
    Function for plotting arrays of images. Not intended to be used directly. See 'images' for typical use cases.
    """
    if n_images > 128:
        raise RuntimeError('The number of subplots exceeds reasonable limits ({})!'.format(n_images))                            
    
    if ncols == 0:
        ncols = int(np.ceil(np.sqrt(n_images)))
    elif ncols < 0:
        ncols = n_images // abs(ncols)

    subplot_x = ncols
    subplot_y = int(np.ceil(n_images / subplot_x))

    if rowlabels is not None and len(rowlabels) != subplot_y:
        raise ValueError('The number of rows does not match the provided labels!')
            
    if titles is not None and type(titles) is str:
        titles = [titles for x in range(n_images)]
        
    if titles is not None and len(titles) != n_images:
        raise ValueError('Provided titles ({}) do not match the number of images ({})!'.format(len(titles), n_images))

    fig = fig or Figure(figsize=(figwidth * subplot_x, figwidth * subplot_y))

    for n in range(n_images):
        ax = fig.add_subplot(subplot_y, subplot_x, n + 1)
        image(fetch_hook(img, n), titles[n] if titles is not None else None, axes=ax, cmap=cmap)
        if rowlabels is not None and n % subplot_x == 0:
            ax.set_ylabel(rowlabels[n // subplot_x])

    return fig
    

def images(imgs, titles=None, figwidth=4, cmap='gray', ncols=0, fig=None, rowlabels=None):
    """
    Plot a series of images (in various structures). Not thoroughly tested, but should work with:

    - np.ndarray of size (h,w,3) or (h,w)
    - lists or tuples of np.ndarray of size (h,w,3) or (h,w)    
    - np.ndarray of size (h,w,channels) -> channels shown separately
    - np.ndarray of size (1, h, w, channels)
    - np.ndarray of size (N, h, w, 3) and (N, h, w, 1)
    
    CAUTION: This function creates new figures without a canvas manager - this prevents memory leaks but makes it more
    difficult to plot figures in interactive notebooks. In case of trouble, you can supply a target figure created by
    a managed matplotlib interface - use fig=plt.figure().

    :param imgs: input image structure (see details above)
    :param titles: a single string or a list of strings matching the number of images in the structure
    :param figwidth: width of a single image in the figure
    :param cmap: color map
    :param ncols: number of columns or: 0 for sqrt(#images) cols; use negative to set the number of rows
    :param fig: specify the target figure for plotting
    """
        
    if type(imgs) is list or type(imgs) is tuple:
        
        n_images = len(imgs)
        
        def fetch_example(image, n):
            return image[n]        
                    
        return _imarray(imgs, n_images, fetch_example, titles, figwidth, cmap, ncols, fig, rowlabels)
            
    elif type(imgs) in [np.ndarray, imageio.core.util.Image]:
        
        if imgs.ndim == 2 or (imgs.ndim == 3 and imgs.shape[-1] == 3):
            
            fig = fig or Figure(tight_layout=True, figsize=(figwidth, figwidth))
            image(imgs, titles, axes=fig.gca(), cmap=cmap)
            
            return fig

        elif imgs.ndim == 3 and imgs.shape[-1] != 3:
                        
            def fetch_example(im, n):
                return im[..., n]
            
            n_images = imgs.shape[-1]

            if n_images > 100:
                imgs = np.moveaxis(imgs, 0, -1)
                n_images = imgs.shape[-1]
                                        
        elif imgs.ndim == 4 and (imgs.shape[-1] == 3 or imgs.shape[-1] == 1):
            
            n_images = imgs.shape[0]
            
            def fetch_example(im, n):
                return im[n]
            
        elif imgs.ndim == 4 and imgs.shape[0] == 1:

            n_images = imgs.shape[-1]
            
            def fetch_example(im, n):
                return im[..., n]

        else:
            raise ValueError('Unsupported array dimensions {}!'.format(imgs.shape))
            
        return _imarray(imgs, n_images, fetch_example, titles, figwidth, cmap, ncols, fig, rowlabels)
            
    else:
        raise ValueError('Unsupported array type {}!'.format(type(imgs)))
                
    return fig


def image(x, label=None, *, axes=None, cmap='gray'):
    """
    Plot a single image, hide ticks & add a formatted title with patterns replaced as follows:
    - '()' -> '(height x width)'
    - '[]' -> '[min - max]'
    - '{}' -> '(height x width) / [min - max]'
    - '<>' -> 'avg ± std'
    """
    
    label = label if label is not None else '{}'
    
    x = x.squeeze()
    
    if any(ptn in label for ptn in ['{}', '()', '[]', '<>']):
        label = label.replace('{}', '() / []')
        label = label.replace('()', '({}x{})'.format(*x.shape[0:2]))
        label = label.replace('[]', '[{:.2f} - {:.2f}]'.format(np.min(x), np.max(x)))
        label = label.replace('<>', '{:.2f} ± {:.2f}'.format(np.mean(x), np.std(x)))
        
    if axes is None:
        fig = Figure()
        axes = fig.gca()

    axes.imshow(x, cmap=cmap)
    if len(label) > 0:
        axes.set_title(label)
    axes.set_xticks([])
    axes.set_yticks([])

    if 'fig' in locals():
        return fig


def sub(n_plots, figwidth=6, figheight=None, ncols=-1, fig=None, transpose=False):
    """
    Create a figure and split it into subplots. Provides more consistent behavior than matplotlib. Key features:
    - the returned axes are always a list
    - automatically choose number of rows/columns based on the total number of plots
    - the extra subplots will be turned off
    - axes traversal order can be changed (column/row-wise)

    :param n_plots:
    :param figwidth:
    :param figheight:
    :param ncols:
    :param fig:
    :param transpose:
    :return:
    """

    if ncols == 0:
        ncols = int(np.ceil(np.sqrt(n_plots)))
    elif ncols < 0:
        ncols = n_plots // abs(ncols)

    figheight = figheight or figwidth

    subplot_x = ncols or int(np.ceil(np.sqrt(n_plots)))
    subplot_y = int(np.ceil(n_plots / subplot_x))

    if transpose:
        subplot_x, subplot_y = subplot_y, subplot_x

    fig = fig or Figure(tight_layout=True, figsize=(figwidth * subplot_x, subplot_y * (figheight or figwidth * (subplot_y / subplot_x))))
    axes = fig.subplots(nrows=subplot_y, ncols=subplot_x)
    axes_flat = []

    if not hasattr(axes, '__iter__'):
        axes = [axes]

    for ax in axes:
        
        if hasattr(ax, '__iter__'):
            for a in ax:
                if len(axes_flat) < n_plots:
                    axes_flat.append(a)
                else:
                    a.remove()
        else:
            if len(axes_flat) < n_plots:
                axes_flat.append(ax)
            else:
                ax.remove()

    if transpose:
        from itertools import product
        axes_flat = [axes_flat[j * subplot_x + i] for i, j in product(range(subplot_x), range(subplot_y))]
    
    return fig, axes_flat


def progress(k, v, results=('training', 'validation'), log='auto', axes=None, start=0, alpha=0.8):
    active = False
    markers = '.os^'[:len(results)]
    for ri, r in enumerate(results):
        if r not in v or len(v[r]) == 0:
            continue
        n_hist = len(v[r]) // 2
        active = True
        xr = start + np.linspace(0, 100, len(v[r]))
        axes.set_title(k)
        axes.plot(xr, v[r], f'C{ri}{markers[ri]}', alpha=0.5)
        axes.plot(xr, stats.ma_exp(v[r], alpha), f'C{ri}-', label='{} ({:.3f})'.format(r, v[r][-1]))
        if (log == 'auto' and np.std(v[r][-n_hist:])/(max(v[r]) - min(v[r])) < 0.02) or (isinstance(log, bool) and log):
            axes.set_yscale('log')
        axes.set_xlabel('Training progress [%]')
    if active: axes.legend()


def perf(training_progress, results=None, figwidth=5, log='auto', fig=None, alpha=0.25):
    """
    Plots training performance stats organized into a dictionary with the following structure:
     - {metric}/{training,validation} -> [values]
     - {metric} -> [values]

    :param training_progress: dictionary with training progress
    :param results: tuple or string, specifies which results to show, e.g., ('training', 'validation') or 'training'
    :param figwidth: width of a single subplot
    :param log: whether to use log scale
    :param fig: handle to matplotlib figure
    :param alpha: parameter for the exponential moving average
    :return: figure handle
    """

    if isinstance(results, str):
        results = (results, )

    # If the data is not formatted as {metric: {training: [values], validation: [values]}} but rather {metric: [values]}
    # convert to the expected structure
    if any(not isinstance(v, dict) for v in training_progress.values()):
        training_progress = {k: v for k, v in training_progress.items() if isinstance(v, dict)}
        training_progress.update({k: {'auto': v} for k, v in training_progress.items() if utils.is_vector(v)})

    # Find the number of metrics with available data
    active = []

    for i, (k, v) in enumerate(training_progress.items()):
        # Check if all training metrics have all requested sets of results
        if results is None or all(r in v and len(v[r]) > 0 for r in results):
            active.append(k)

    if len(active) == 0:
        raise ValueError('No valid plots! Missing training/validation data? Use results=["training"] to select.')

    fig, axes = sub(len(active), ncols=-1, fig=fig)
    fig.set_size_inches((len(active) * figwidth, figwidth * 0.75))

    for i, k in enumerate(active):
        v = training_progress[k]
        progress(k, v, results or v.keys(), log, axes[i], alpha=alpha)
    
    return fig


def detection(positive, negative, bins=200, axes=None, title='()', scale=True, reference=None, guides=2):
    """
    Plot histograms of positive & negative detection scores.

    :param positive: positive detection scores (numpy array)
    :param negative: positive detection scores (numpy array)
    :param bins: number of histogram bins
    :param axes: matplotlib axes' handle
    :param title: plot title, '()' will be replaced with accuracy and tpr stats
    :param scale: boolean flag to auto select x limits
    :param reference: additional scores to be plotted as a reference (shown in gray)
    :param guides: draw lines as guides: 0 (no lines), 1 (best accuracy threshold), 2 (threshold + percentiles)
    :return: figure handle (if created here)
    """

    cc_min = np.min([positive.min(), negative.min()])
    cc_max = np.max([positive.max(), negative.max()])

    if reference is not None:
        cc_max = np.max([cc_max, reference.max()])
        cc_min = np.min([cc_min, reference.min()])

    cc = np.linspace(cc_min, cc_max, bins)
    bin_accuracy, thr_id = stats.detection_accuracy(positive, negative, cc, return_index=True)
    tpr = stats.true_positive_rate(positive, negative)

    v_no_match_min = np.percentile(negative, 99)
    v_do_match_max = np.percentile(positive, 1)

    if axes is None:
        fig = Figure()
        axes = fig.gca()

    # From bin centers, convert to bin edges for histogram computation
    h_bins = stats.bin_edges(cc)
    
    h1 = axes.hist(positive.ravel(), h_bins, color='g', alpha=0.4, density=True, label='positive')
    h2 = axes.hist(negative.ravel(), h_bins, color='r', alpha=0.4, density=True, label='negative')
    if reference is not None:
        h3 = axes.hist(reference.ravel(), h_bins, color='gray', alpha=0.4, density=True, label='reference')

    h_max = 1.05 * max( np.max(h1[0]), np.max(h2[0]) )
    if guides == 2:
        axes.plot([v_do_match_max, v_do_match_max], [0, h_max], 'g--')
        axes.plot([v_no_match_min, v_no_match_min], [0, h_max], 'r--')
        axes.plot([cc[thr_id], cc[thr_id]], [0, max(h1[0][thr_id], 0.05 * h_max)], 'k:')
    elif guides == 1:
        axes.plot([cc[thr_id], cc[thr_id]], [0, h_max], 'k:')
    
    axes.set_title(title.replace('()', f'acc. {bin_accuracy:.2f}, tpr @ 1\\%far={tpr:.2f}'))
    
    if scale:
        axes.set_xlim([1.1 * cc_min if cc_min < 0 else 0.9 * cc_min, 1.1 * cc_max])
    
    axes.set_ylim([0, h_max])
    axes.legend()

    if 'fig' in locals():
        return fig


def roc(matching, non_matching, bins=100, axes=None, label=None, plot_guides=True):

    tpr, fpr = stats.roc(matching, non_matching, bins)
    tpr_at_1pp_fpr = stats.true_positive_rate(matching, non_matching, 0.01)
    auc = stats.auc(matching, non_matching)

    if axes is None:
        fig = Figure()
        axes = fig.gca()

    label = f'{label} : tpr={tpr_at_1pp_fpr:.2f} auc={auc:.2f}' if label is not None else None
    axes.plot(fpr, tpr, '-', label=label)
    if plot_guides:
        axes.plot([0, 1], [0, 1], 'k--', alpha=0.2)
        axes.plot([0.01, 0.01], [0, tpr_at_1pp_fpr], 'k:', alpha=0.25)
        axes.plot([0.01, 1], [tpr_at_1pp_fpr, tpr_at_1pp_fpr], 'k:', alpha=0.25)
    axes.set_xlim([-0.02, 1.02])
    axes.set_ylim([-0.02, 1.02])
    axes.set_xlabel('false positive rate')
    axes.set_ylabel('true positive rate')
    if label is not None: axes.legend()

    if 'fig' in locals():
        return fig


def intervals_bulk(x, y, p=10):
    # fig, axes = plt.subplots(nrows=1, ncols=len(y), sharex=True)
    fig, axes = sub(len(y), ncols=-1)
    fig.set_size_inches((6 * len(y), 3))
    
    xl = sorted(x.keys())[0]
    xv = x[xl]
    
    for i, (k, v) in enumerate(y.items()):
        axes[i].plot(xv, np.percentile(v, 50, axis=0))
        axes[i].fill_between(xv, np.percentile(v, p, axis=0), np.percentile(v, 100-p, axis=0), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',)
        axes[i].set_ylabel(k)
        axes[i].set_xlabel(xl)

    return fig


def intervals(x, y, p=10, xlabel=None, ylabel=None, style='.-', axes=None):
    axes.plot(x, np.percentile(y, 50, axis=0), style)
    axes.fill_between(x, np.percentile(y, p, axis=0), np.percentile(y, 100-p, axis=0), alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',)
    if ylabel is not None: axes.set_ylabel(ylabel)
    if xlabel is not None: axes.set_xlabel(xlabel)


def correlation(x, y, xlabel=None, ylabel=None, title=None, axes=None, alpha=0.1, guide=False):

    title = '{} : '.format(title) if title is not None else ''

    cc = stats.corrcoeff(x.ravel(), y.ravel())
    r2 = stats.rsquared(x.ravel(), y.ravel())

    axes.plot(x.ravel(), y.ravel(), '.', alpha=alpha)
    axes.set_title('{}corr {:.2f} / R2 {:.2f}'.format(title, cc, r2))

    if guide:
        axes.plot(axes.get_xlim(), axes.get_ylim(), 'k:', alpha=0.2)

    if xlabel is not None: axes.set_xlabel(xlabel)
    if ylabel is not None: axes.set_ylabel(ylabel)


def scatter_hex(x, y, xlabel=None, ylabel=None, axes=None, marginals=True, bins=30):
    axes.hexbin(x, y, gridsize=50, bins=bins, cmap='Blues')
    axes.set_xticks([])
    axes.set_yticks([])

    if xlabel is not None: axes.set_xlabel(xlabel)
    if ylabel is not None: axes.set_ylabel(ylabel)

    if marginals:
        # X marginal
        x_hist, x_bins = np.histogram(x.reshape((-1, )), bins=bins)
        x_bins = np.convolve(x_bins, [0.5, 0.5], mode='valid')
        x_hist = x_hist / x_hist.max()
        yy = axes.get_ylim()
        axes.bar(x_bins, bottom=yy[1], height=0.1 * np.abs(yy[1] - yy[0]) * x_hist, zorder=-1, clip_on=False, alpha=0.5, width=x_bins[1] - x_bins[0])
        axes.set_ylim(yy)

        # Y marginal
        y_hist, y_bins = np.histogram(y.reshape((-1, )), bins=bins)
        y_bins = np.convolve(y_bins, [0.5, 0.5], mode='valid')
        y_hist = y_hist / y_hist.max()
        xx = axes.get_xlim()
        axes.barh(y_bins, left=xx[1], width=0.1 * np.abs(xx[1] - xx[0]) * y_hist, zorder=-1, clip_on=False, alpha=0.5, height=y_bins[1] - y_bins[0])
        axes.set_xlim(xx)

