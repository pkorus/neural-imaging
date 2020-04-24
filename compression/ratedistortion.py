import os
import tqdm
import imageio
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glymur
from pathlib import Path
from skimage.measure import compare_ssim, compare_psnr
from sewar.full_ref import msssim

from scipy.optimize import curve_fit

import helpers.stats
import helpers.utils
from helpers import loading, utils, fsutil
from compression import jpeg_helpers, codec, bpg_helpers

from loguru import logger


def get_jpeg_df(directory, write_files=False, effective_bytes=True, force_calc=False):
    """
    Compute and return (as Pandas DF) the rate distortion curve for JPEG. The result is saved
    as a CSV file in the source directory. If the file exists, the DF is loaded and returned.

    Files are saved as JPEG using imageio.
    """

    files, _ = loading.discover_images(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    # Get trade-off for JPEG
    quality_levels = np.arange(95, 5, -5)
    df_jpeg_path = os.path.join(directory, 'jpeg.csv')

    if os.path.isfile(df_jpeg_path) and not force_calc:
        logger.info('Restoring JPEG stats from {}'.format(df_jpeg_path))
        df = pd.read_csv(df_jpeg_path, index_col=False)
    else:
        df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'msssim', 'msssim_db', 'bytes', 'bpp'])

        with tqdm.tqdm(total=len(files) * len(quality_levels), ncols=120, desc='JPEG') as pbar:

            for image_id, filename in enumerate(files):

                # Read the original image
                image = batch_x[image_id]

                for qi, q in enumerate(quality_levels):

                    # Compress images and get effective bytes (only image data - no headers)
                    image_compressed, image_bytes = jpeg_helpers.compress_batch(image, q, effective=effective_bytes)

                    if write_files:
                        image_dir = os.path.join(directory, os.path.splitext(filename)[0])
                        if not os.path.isdir(image_dir):
                            os.makedirs(image_dir)

                        image_path = os.path.join(image_dir, 'jpeg_q{:03d}.png'.format(q))
                        imageio.imwrite(image_path, (255 * image_compressed).astype(np.uint8))

                    msssim_value = msssim(image, image_compressed, MAX=1).real

                    df = df.append({'image_id': image_id,
                                    'filename': filename,
                                    'codec': 'jpeg',
                                    'quality': q,
                                    'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                    'psnr': compare_psnr(image, image_compressed, data_range=1),
                                    'msssim': msssim_value,
                                    'msssim_db': -10 * np.log10(1 - msssim_value),
                                    'bytes': image_bytes,
                                    'bpp': 8 * image_bytes / image.shape[0] / image.shape[1]
                                    }, ignore_index=True)

                    pbar.set_postfix(image_id=image_id, quality=q)
                    pbar.update(1)

        df.to_csv(os.path.join(directory, 'jpeg.csv'), index=False)

    return df


def get_jpeg2k_df(directory, write_files=False, effective_bytes=True, force_calc=False):
    """
    Compute and return (as Pandas DF) the rate distortion curve for JPEG 2000. The result is saved
    as a CSV file in the source directory. If the file exists, the DF is loaded and returned.

    Files are saved as JPEG using glymur.
    """

    files, _ = loading.discover_images(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    # Get trade-off for JPEG
    quality_levels = np.arange(25, 45, 1)
    df_jpeg_path = os.path.join(directory, 'jpeg2000.csv')

    if os.path.isfile(df_jpeg_path) and not force_calc:
        logger.info('Restoring JPEG 2000 stats from {}'.format(df_jpeg_path))
        df = pd.read_csv(df_jpeg_path, index_col=False)
    else:
        df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'msssim', 'msssim_db', 'bytes', 'bpp'])

        with tqdm.tqdm(total=len(files) * len(quality_levels), ncols=120, desc='JP2k') as pbar:

            for image_id, filename in enumerate(files):

                # Read the original image
                image = batch_x[image_id]

                for qi, q in enumerate(quality_levels):

                    # TODO Use Glymur to save JPEG 2000 images to a temp file
                    image_np = (255 * image.clip(0, 1)).astype(np.uint8)
                    glymur.Jp2k('/tmp/image.jp2', data=image_np, psnr=[q])
                    if effective_bytes:
                        image_bytes = jpeg_helpers.jp2bytes('/tmp/image.jp2')
                    else:
                        image_bytes = os.path.getsize('/tmp/image.jp2')
                    image_compressed = imageio.imread('/tmp/image.jp2').astype(np.float) / (2**8 - 1)

                    # TODO Use Pillow to save JPEG 2000 images to a memory buffer
                    # TODO This has been disabled = their implementation seems to be invalid
                    # with io.BytesIO() as output:
                    #     image_pillow = PIL.Image.fromarray((255*image.clip(0, 1)).astype(np.uint8))
                    #     image_pillow.save(output, format='jpeg2000', quality_layers=[q])
                    #     image_compressed = imageio.imread(output.getvalue()).astype(np.float) / (2**8 - 1)
                    #     image_bytes = len(output.getvalue())

                    if write_files:
                        image_dir = os.path.join(directory, os.path.splitext(filename)[0])
                        if not os.path.isdir(image_dir):
                            os.makedirs(image_dir)

                        image_path = os.path.join(image_dir, 'jp2_q{:.1f}dB.png'.format(q))
                        imageio.imwrite(image_path, (255*image_compressed).astype(np.uint8))

                    msssim_value = msssim(image, image_compressed, MAX=1).real

                    df = df.append({'image_id': image_id,
                                    'filename': filename,
                                    'codec': 'jpeg2000',
                                    'quality': q,
                                    'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                    'psnr': compare_psnr(image, image_compressed, data_range=1),
                                    'msssim': msssim_value,
                                    'msssim_db': -10 * np.log10(1 - msssim_value),
                                    'bytes': image_bytes,
                                    'bpp': 8 * image_bytes / image.shape[0] / image.shape[1]
                                    }, ignore_index=True)

                    pbar.set_postfix(image_id=image_id, quality=q)
                    pbar.update(1)

        df.to_csv(df_jpeg_path, index=False)

    return df


def get_bpg_df(directory, write_files=False, effective_bytes=True, force_calc=False):
    """
    Compute and return (as Pandas DF) the rate distortion curve for BPG. The result is saved
    as a CSV file in the source directory. If the file exists, the DF is loaded and returned.

    The files are saved using the reference codec: https://bellard.org/bpg/
    """

    files, _ = loading.discover_images(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astypre(np.float32) / (2 ** 8 - 1)

    quality_levels = np.arange(10, 40, 1)
    df_jpeg_path = os.path.join(directory, 'bpg.csv')

    if os.path.isfile(df_jpeg_path) and not force_calc:
        logger.info('Restoring BPG stats from {}'.format(df_jpeg_path))
        df = pd.read_csv(df_jpeg_path, index_col=False)
    else:
        df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'msssim', 'msssim_db', 'bytes', 'bpp'])

        with tqdm.tqdm(total=len(files) * len(quality_levels), ncols=120, desc='BPG') as pbar:

            for image_id, filename in enumerate(files):

                # Read the original image
                image = batch_x[image_id]

                for qi, q in enumerate(quality_levels):

                    # Compress to BPG
                    # Save as temporary file
                    imageio.imwrite('/tmp/image.png', (255*image).astype(np.uint8))
                    bpp_path = bpg_helpers.bpg_compress('/tmp/image.png', q, '/tmp')
                    image_compressed = imageio.imread(bpg_helpers.decode_bpg_to_png(bpp_path)).astype(np.float) / (2**8 - 1)
                    
                    if effective_bytes:
                        bpp = bpg_helpers.bpp_of_bpg_image(bpp_path)
                        image_bytes = round(bpp * image.shape[0] * image.shape[1] / 8)
                    else:
                        image_bytes = os.stat(bpp_path).st_size
                        bpp = 8 * image_bytes / image.shape[0] / image.shape[1]

                    if write_files:
                        image_dir = os.path.join(directory, os.path.splitext(filename)[0])
                        if not os.path.isdir(image_dir):
                            os.makedirs(image_dir)

                        image_path = os.path.join(image_dir, 'bpg_q{:03d}.png'.format(q))
                        imageio.imwrite(image_path, (255 * image_compressed).astype(np.uint8))

                    msssim_value = msssim(image, image_compressed, MAX=1).real

                    df = df.append({'image_id': image_id,
                                    'filename': filename,
                                    'codec': 'bpg',
                                    'quality': q,
                                    'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                    'psnr': compare_psnr(image, image_compressed, data_range=1),
                                    'msssim': msssim_value,
                                    'msssim_db': -10 * np.log10(1 - msssim_value),
                                    'bytes': image_bytes,
                                    'bpp': bpp
                                    }, ignore_index=True)

                    pbar.set_postfix(image_id=image_id, quality=q)
                    pbar.update(1)

        df.to_csv(df_jpeg_path, index=False)

    return df


def get_dcn_df(directory, model_directory, write_files=False, force_calc=False):
    """
    Compute and return (as Pandas DF) the rate distortion curve for the learned DCN codec.
    The result is saved as a CSV file in the source directory. If the file exists, the DF
    is loaded and returned.
    """

    # Discover test files
    files, _ = loading.discover_images(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    # Create a new table for the DCN
    df = pd.DataFrame(
        columns=['image_id', 'filename', 'model_dir', 'codec', 'ssim', 'psnr', 'msssim', 'msssim_db', 'entropy', 'bytes', 'bpp', 'layers', 'quantization', 'entropy_reg', 'codebook', 'latent', 'latent_shape', 'n_features'])

    # Discover available models
    model_dirs = list(Path(model_directory).glob('**/progress.json'))
    logger.info('Found {} models'.format(len(model_dirs)))

    df_path = os.path.join(directory, 'dcn-{}.csv'.format([x for x in fsutil.split(model_directory) if len(x) > 0][-1]))

    if os.path.isfile(df_path) and not force_calc:
        logger.info('Restoring DCN stats from {}'.format(df_path))
        df = pd.read_csv(df_path, index_col=False)
    else:

        for model_dir in model_dirs:
            logger.info('Processing model dir: {}'.format(model_dir))
            dcn = codec.restore(os.path.split(str(model_dir))[0], batch_x.shape[1])

            # Dump compressed images
            for image_id, filename in enumerate(files):

                try:
                    batch_y, image_bytes = codec.simulate_compression(batch_x[image_id:image_id + 1], dcn)
                    batch_z = dcn.compress(batch_x[image_id:image_id + 1])
                    entropy = helpers.stats.entropy(batch_z, dcn.get_codebook())
                except Exception as e:
                    logger.error('Error while processing {} with {} : {}'.format(filename, dcn.model_code, e))
                    raise e

                if write_files:
                    image_dir = os.path.join(directory, os.path.splitext(filename)[0])
                    if not os.path.isdir(image_dir):
                        os.makedirs(image_dir)

                    image_path = os.path.join(image_dir, dcn.model_code.replace('/', '-') + '.png')
                    imageio.imwrite(image_path, (255 * batch_y[0]).astype(np.uint8))

                msssim_value = msssim(batch_x[image_id], batch_y[0], MAX=1).real

                df = df.append({'image_id': image_id,
                                'filename': filename,
                                'model_dir': os.path.relpath(os.path.split(str(model_dir))[0], model_directory).replace(dcn.scoped_name, ''),
                                'codec': dcn.model_code,
                                'ssim': compare_ssim(batch_x[image_id], batch_y[0], multichannel=True, data_range=1),
                                'psnr': compare_psnr(batch_x[image_id], batch_y[0], data_range=1),
                                'msssim': msssim_value,
                                'msssim_db': -10 * np.log10(1 - msssim_value),
                                'entropy': entropy,
                                'bytes': image_bytes,
                                'bpp': 8 * image_bytes / batch_x[image_id].shape[0] / batch_x[image_id].shape[1],
                                'layers': dcn.n_layers if 'n_layers' in dcn._h else None,
                                'quantization': '{}-{:.0f}bpf'.format(dcn._h.rounding, dcn.latent_bpf),
                                'entropy_reg': dcn.entropy_weight,
                                'codebook': dcn._h.rounding,
                                'latent': dcn.n_latent,
                                'latent_shape': '{}x{}x{}'.format(*dcn.latent_shape[1:]),
                                'n_features': dcn.latent_shape[-1]
                                }, ignore_index=True)

        df.to_csv(df_path, index=False)

    return df


def load_data(plots, dirname):
    """
    Returns data frames with numerical results for specified codecs [and settings]

    Example definition (can be both a list or a dictionary):

    plots = OrderedDict()
    plots['jpg'] = ('jpeg.csv', {})
    plots['jp2'] = ('jpeg2000.csv', {})
    plots['bpg'] = ('bpg.csv', {})
    plots['dcn'] = ('dcn-7-raw.csv', {'model_dir': '.*basic/'})

    Tuple structure: (filename, data filtering conditions - dict {column: value})

    """

    # Load all needed tables and setup legend labels
    labels = []
    df_all = []

    if isinstance(plots, list):
        for filename, selectors in plots:
            labels.append(os.path.splitext(filename)[0])
            df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
            for k, v in selectors.items():
                if isinstance(v, str) and '*' in v:
                    df = df[df[k].str.match(v)]
                else:
                    df = df[df[k] == v]
            if len(df) == 0:
                raise (ValueError('No rows matched for column {}'.format(k)))
            df_all.append(df)

    elif isinstance(plots, dict):
        for key, (filename, selectors) in plots.items():
            labels.append(key)
            df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
            for k, v in selectors.items():
                if isinstance(v, str) and '*' in v:
                    df = df[df[k].str.match(v)]
                else:
                    df = df[df[k] == v]
            if len(df) == 0:
                raise (ValueError('No rows matched for column {}'.format(k)))
            df_all.append(df)
    else:
        raise ValueError('Unsupported plot definition!')

    return df_all, labels


def setup_plot(metric):
    if metric == 'psnr':
        y_min = 25
        y_max = 45
        metric_label = 'PSNR [dB]'

    elif metric == 'msssim_db':
        y_min = 10
        y_max = 32
        metric_label = 'MS-SSIM [dB]'

    elif metric == 'ssim':
        y_min = 0.8
        y_max = 1
        metric_label = 'SSIM'

    elif metric == 'msssim':
        y_min = 0.9
        y_max = 1
        metric_label = 'MS-SSIM'
    else:
        raise ValueError('Unsupported metric!')

    return y_min, y_max, metric_label


def setup_fit(metric):
    # Define a parametric model for the trade-off curve
    if metric in {'ssim', 'msssim'}:
        # These bounds work well for baseline fitting
        fit_bounds = ([1e-4, 1e-2, -3, -0.5], [5, 15, 5, 0.5])
        # These bounds work better for optimized DCN codecs - there are some weird outliers in the data
        # fit_bounds = ([0.1, 1e-5, -1, 0], [3, 10, 7, 0.1])

        def func(x, a, b, c, d):
            return 1 / (1 + np.exp(- b * x ** a + c)) - d
    else:
        # These bounds work well for baseline fitting
        fit_bounds = ([1e-4, 1e-5, 1e-2, -50], [100, 100, 3, 50])
        # These bounds work better for optimized DCN codecs - there are some weird outliers in the data
        # fit_bounds = ([1e-4, 1, 1e-2, -20], [20, 50, 1, 20])

        def func(x, a, b, c, d):
            return a * np.log(np.clip(b * x ** c + d, a_min=1e-9, a_max=1e9))

    return func, fit_bounds


def plot_curve(plots, axes,
               dirname='./data/rgb/clic256',
               images=[],
               plot='fit',
               draw_markers=None,
               metric='ssim',
               title=None,
               add_legend=True,
               marker_legend=True,
               baseline_count=3,
               update_ylim=False):

    # Parse input parameters
    draw_markers = draw_markers if draw_markers is not None else len(images) == 1
    plot = helpers.utils.match_option(plot, ['fit', 'aggregate'])

    df_all, labels = load_data(plots, dirname)

    if len(images) == 0:
        images = df_all[0]['image_id'].unique().tolist()

    # Plot setup
    func, fit_bounds = setup_fit(metric)
    y_min, y_max, metric_label = setup_plot(metric)

    # Select measurements for specific images, if specified
    for dfc in df_all:
        if len(images) > 0:
            dfc['selected'] = dfc['image_id'].apply(lambda x: x in images)
        else:
            dfc['selected'] = True

    # Setup drawing styles
    styles = [['r-', 'rx'], ['b--', 'b+'], ['k:', 'k2'], ['g-', 'gx'], ['m-', 'gx'], ['m--', 'gx'], ['m-.', 'gx'], ['m:', 'gx']]
    avg_markers = ['', '', '', 'o', 'o', '2', '+', 'x', '^', '.']

    # To retain consistent styles across plots, adjust the lists based on the number of baseline methods
    if baseline_count < 3:
        styles = styles[(3 - baseline_count):]
        avg_markers = avg_markers[(3 - baseline_count):]

    # Iterate over defined plots and draw data accordingly
    for index, dfc in enumerate(df_all):

        x = dfc.loc[dfc['selected'], 'bpp'].values
        y = dfc.loc[dfc['selected'], metric].values

        X = np.linspace(max([0, x.min() * 0.9]), min([5, x.max() * 1.1]), 256)

        if plot == 'fit':
            # Fit individual images to a curve, then average the curves

            Y = np.zeros((len(images), len(X)))
            mse_l = []

            for image_no, image_id in enumerate(images):

                x = dfc.loc[dfc['selected'] & (dfc['image_id'] == image_id), 'bpp'].values
                y = dfc.loc[dfc['selected'] & (dfc['image_id'] == image_id), metric].values

                # Allow for larger errors for lower SSIM values
                if metric in ['ssim', 'msssim']:
                    sigma = np.abs(1 - y).reshape((-1,))
                else:
                    sigma = np.ones_like(y).reshape((-1,))

                try:
                    popt, pcov = curve_fit(func, x, y, bounds=fit_bounds, maxfev=10000, sigma=sigma)
                    y_est = func(x, *popt)
                    mse = np.mean(np.power(y - y_est, 2))
                    mse_l.append(mse)
                    if mse > 0.5:
                        logger.warning('WARNING Large MSE for {}:{} = {:.2f}'.format(labels[index], image_no, mse))

                except RuntimeError:
                    logger.error(f'{labels[index]} image ={image_id}, bpp ={x} y ={y}')

                Y[image_no] = func(X, *popt)

            if len(images) > 1:
                logger.info('Fit summary - MSE for {} av={:.2f} max={:.2f}'.format(labels[index], np.mean(mse_l), np.max(mse_l)))

            yy = np.nanmean(Y, axis=0)
            axes.plot(X, yy, styles[index][0], label=labels[index] if add_legend else None)
            y_min = min([y_min, min(yy)]) if update_ylim else y_min

        elif plot == 'aggregate':
            # For each quality level (QF, #channels) find the average quality level
            dfa = dfc.loc[dfc['selected']]

            if 'n_features' in dfa:
                dfg = dfa.groupby('n_features')
            else:
                dfg = dfa.groupby('quality')

            x = dfg.mean()['bpp'].values
            y = dfg.mean()[metric].values

            axes.plot(x, y, styles[index][0], label=labels[index] if add_legend else None, marker=avg_markers[index], alpha=0.65)
            y_min = min([y_min, min(y)]) if update_ylim else y_min

        elif plot == 'none':
            pass

        else:
            raise ValueError('Unsupported plot type!')

        if draw_markers:

            if 'entropy_reg' in dfc:

                # No need to draw legend if multiple DCNs are plotted
                detailed_legend = 'full' if marker_legend and index == baseline_count else False

                style_mapping = {}

                if 'n_features' in dfc and len(dfc['n_features'].unique()) > 1:
                    style_mapping['hue'] = 'n_features'

                if 'entropy_reg' in dfc and len(dfc['entropy_reg'].unique()) > 1:
                    style_mapping['size'] = 'entropy_reg'

                if 'quantization' in dfc and len(dfc['quantization'].unique()) > 1:
                    style_mapping['style'] = 'quantization'

                sns.scatterplot(data=dfc[dfc['selected']], x='bpp', y=metric,
                                palette="Set2", ax=axes, legend=detailed_legend,
                                **style_mapping)

            else:
                axes.plot(x, y, styles[index][1], alpha=10 / (sum(dfc['selected'])))

    n_images = len(dfc.loc[dfc['selected'], 'image_id'].unique())

    title = '{} : {}'.format(
        title if title is not None else os.path.split(dirname)[-1],
        '{} images'.format(n_images) if n_images > 1 else dfc.loc[dfc['selected'], 'filename'].unique()[0].replace('.png', '')
    )

    # Fixes problems with rendering using the LaTeX backend
    if add_legend:
        for t in axes.legend().texts:
            t.set_text(t.get_text().replace('_', '-'))

    axes.set_xlim([-0.1, 3.1])
    axes.set_ylim([y_min * 0.99, y_max])
    axes.set_title(title)
    axes.set_xlabel('Effective bpp')
    axes.set_ylabel(metric_label)


def plot_bulk(plots, dirname, plot_images, metric, plot, baseline_count=3, add_legend=True, max_bpp=5,
              draw_markers=1):
    plot = helpers.utils.match_option(plot, ['fit', 'aggregate'])
    if dirname.endswith('/') or dirname.endswith('\\'):
        dirname = dirname[:-1]

    # Load data and select images for plotting
    df_all, labels = load_data(plots, dirname)
    plot_images = plot_images if len(plot_images) > 0 else [-1] + df_all[0].image_id.unique().tolist()
    logger.info(f'Selected images: {plot_images}')

    images_x = int(np.ceil(np.sqrt(len(plot_images))))
    images_y = int(np.ceil(len(plot_images) / images_x))

    update_ylim = False
    marker_legend = False

    # Plot setup
    func, fit_bounds = setup_fit(metric)
    y_min, y_max, metric_label = setup_plot(metric)

    # Setup drawing styles
    styles = [['r-', 'rx'], ['b--', 'b+'], ['k:', 'k2'], ['g-', 'gx'], ['m-', 'gx'], ['m--', 'gx'], ['m-.', 'gx'],
              ['m:', 'gx']]
    avg_markers = ['', '', '', 'o', 'o', '2', '+', 'X', '^', '.']

    # To retain consistent styles across plots, adjust the lists based on the number of baseline methods
    if baseline_count < 3:
        styles = styles[(3 - baseline_count):]
        avg_markers = avg_markers[(3 - baseline_count):]

    mse_labels = {}

    fig, ax = plt.subplots(images_y, images_x, sharex=True, sharey=True)
    fig.set_size_inches((images_x * 6, images_y * 4))

    if hasattr(ax, 'flat'):
        for axes in ax.flat:
            axes.axis('off')

    for ax_id, image_id in enumerate(plot_images):

        if images_y > 1:
            axes = ax[ax_id // images_x, ax_id % images_x]
        elif images_x > 1:
            axes = ax[ax_id % images_x]
        else:
            axes = ax

        axes.axis('on')

        # Select measurements for a specific image, if specified
        for dfc in df_all:
            if image_id >= 0:
                dfc['selected'] = dfc['image_id'].apply(lambda x: x == image_id)
            else:
                dfc['selected'] = True

        for index, dfc in enumerate(df_all):

            x = dfc.loc[dfc['selected'], 'bpp'].values
            y = dfc.loc[dfc['selected'], metric].values

            X = np.linspace(max([0, x.min() * 0.9]), min([5, x.max() * 1.1]), 256)

            if plot == 'fit':
                # Fit individual images to a curve, then average the curves

                if image_id >= 0:
                    images = [image_id]
                else:
                    images = dfc.image_id.unique()

                Y = np.zeros((len(images), len(X)))
                mse_l = []

                for image_no, imid in enumerate(images):

                    x = dfc.loc[dfc['selected'] & (dfc['image_id'] == imid), 'bpp'].values
                    y = dfc.loc[dfc['selected'] & (dfc['image_id'] == imid), metric].values

                    # Allow for larger errors for lower SSIM values
                    if metric in ['ssim', 'msssim']:
                        sigma = np.abs(1 - y).reshape((-1,))
                    else:
                        sigma = np.ones_like(y).reshape((-1,))

                    try:
                        popt, pcov = curve_fit(func, x, y, bounds=fit_bounds, sigma=sigma, maxfev=100000)
                        y_est = func(x, *popt)
                        mse = np.mean(np.power(y - y_est, 2))
                        mse_l.append(mse)
                        if mse > 0.1:
                            logger.warning('WARNING Large MSE for {} img=#{} = {:.2f}'.format(labels[index], image_no, mse))

                    except RuntimeError as err:
                        logger.error(f'{labels[index]} image ={imid} bpp={x} y ={y} err ={err}')

                    Y[image_no] = func(X, *popt)

                if image_id < 0:
                    logger.info('Fit summary - MSE for {} av={:.2f} max={:.2f}'.format(labels[index], np.mean(mse_l),
                                                                                 np.max(mse_l)))
                mse_labels[labels[index]] = np.mean(mse_l)

                yy = np.nanmean(Y, axis=0)
                axes.plot(X, yy, styles[index][0],
                          label='{} ({:.3f})'.format(labels[index], mse_labels[labels[index]]) if add_legend else None)
                y_min = min([y_min, min(yy)]) if update_ylim else y_min

            elif plot == 'aggregate':
                # For each quality level (QF, #channels) find the average quality level
                dfa = dfc.loc[dfc['selected']]

                if 'n_features' in dfa:
                    dfg = dfa.groupby('n_features')
                else:
                    dfg = dfa.groupby('quality')

                x = dfg.mean()['bpp'].values
                y = dfg.mean()[metric].values

                axes.plot(x, y, styles[index][0], label=labels[index] if add_legend else None,
                          marker=avg_markers[index], alpha=0.65)
                y_min = min([y_min, min(y)]) if update_ylim else y_min

            elif plot == 'none':
                pass

            else:
                raise ValueError('Unsupported plot type!')

            if draw_markers > 0:

                if 'entropy_reg' in dfc:

                    if image_id >= 0 or draw_markers >= 2:

                        # No need to draw legend if multiple DCNs are plotted
                        detailed_legend = 'full' if marker_legend and index == baseline_count else False

                        style_mapping = {}

                        if 'n_features' in dfc and len(dfc['n_features'].unique()) > 1:
                            style_mapping['hue'] = 'n_features'

                        if 'entropy_reg' in dfc and len(dfc['entropy_reg'].unique()) > 1:
                            style_mapping['size'] = 'entropy_reg'

                        if 'quantization' in dfc and len(dfc['quantization'].unique()) > 1:
                            style_mapping['style'] = 'quantization'

                        sns.scatterplot(data=dfc[dfc['selected']], x='bpp', y=metric,
                                        palette="Set2", ax=axes, legend=detailed_legend,
                                        **style_mapping)

                else:

                    if image_id >= 0:
                        axes.plot(x, y, styles[index][1], alpha=0.65)

        # Setup title
        n_images = len(dfc.loc[dfc['selected'], 'image_id'].unique())
        if n_images > 1:
            title = '{} for {} images ({})'.format(plot, n_images, os.path.split(dirname)[-1])
        else:
            title = '\#{} : {}'.format(image_id, dfc.loc[dfc['selected'], 'filename'].unique()[0].replace('.png', ''))

        # Fixes problems with rendering using the LaTeX backend
        if add_legend:
            for t in axes.legend().texts:
                t.set_text(t.get_text().replace('_', '-'))

        axes.set_xlim([-0.1, max_bpp + 0.1])
        axes.set_ylim([y_min * 0.95, y_max])
        axes.legend(loc='lower right')
        axes.set_title(title)
        if image_id // images_x == images_y - 1:
            axes.set_xlabel('Effective bpp')
        if image_id % images_x == 0:
            axes.set_ylabel(metric_label)

    return fig