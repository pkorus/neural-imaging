import io
import os
import tqdm
import imageio
import pandas as pd
import numpy as np
import seaborn as sns
import glymur
from pathlib import Path
from skimage.measure import compare_ssim, compare_psnr
from sewar.full_ref import msssim, ssim, psnr

# For curve fitting and regression
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp
import uncertainties as unc

from helpers import loading, utils, coreutils
from compression import jpeg_helpers, afi, bpg_helpers


def get_jpeg_df(directory, write_files=False, effective_bytes=True, force_calc=False):

    files, _ = loading.discover_files(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    # Get trade-off for JPEG
    quality_levels = np.arange(95, 5, -5)
    df_jpeg_path = os.path.join(directory, 'jpeg.csv')

    if os.path.isfile(df_jpeg_path) and not force_calc:
        print('Restoring JPEG stats from {}'.format(df_jpeg_path))
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

        files, _ = loading.discover_files(directory, n_images=-1, v_images=0)
        batch_x = loading.load_images(files, directory, load='y')
        batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

        # Get trade-off for JPEG
        quality_levels = np.arange(25, 45, 1)
        df_jpeg_path = os.path.join(directory, 'jpeg2000.csv')

        if os.path.isfile(df_jpeg_path) and not force_calc:
            print('Restoring JPEG 2000 stats from {}'.format(df_jpeg_path))
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
                        # TODO This has been disabled
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

    files, _ = loading.discover_files(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    quality_levels = np.arange(10, 40, 1)
    df_jpeg_path = os.path.join(directory, 'bpg.csv')

    if os.path.isfile(df_jpeg_path) and not force_calc:
        print('Restoring BPG stats from {}'.format(df_jpeg_path))
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

    # Discover test files
    files, _ = loading.discover_files(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    # Create a new table for the DCN
    df = pd.DataFrame(
        columns=['image_id', 'filename', 'model_dir', 'codec', 'ssim', 'psnr', 'msssim', 'msssim_db', 'entropy', 'bytes', 'bpp', 'layers', 'quantization', 'entropy_reg', 'codebook', 'latent', 'latent_shape', 'n_features'])

    # Discover available models
    model_dirs = list(Path(model_directory).glob('**/progress.json'))
    print('Found {} models'.format(len(model_dirs)))

    df_path = os.path.join(directory, 'dcn-{}.csv'.format([x for x in coreutils.splitall(model_directory) if len(x) > 0][-1]))

    if os.path.isfile(df_path) and not force_calc:
        print('Restoring DCN stats from {}'.format(df_path))
        df = pd.read_csv(df_path, index_col=False)
    else:

        for model_dir in model_dirs:
            print('Processing: {}'.format(model_dir))
            dcn = afi.restore_model(os.path.split(str(model_dir))[0], batch_x.shape[1])

            # Dump compressed images
            for image_id, filename in enumerate(files):

                try:
                    batch_y, image_bytes = afi.dcn_simulate_compression(dcn, batch_x[image_id:image_id + 1])
                    batch_z = dcn.compress(batch_x[image_id:image_id + 1])
                    entropy = utils.entropy(batch_z, dcn.get_codebook())
                except Exception as e:
                    print('Error while processing {} with {} : {}'.format(filename, dcn.model_code, e))
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
               dump_df=False,
               update_ylim=False,
               db_scale=False):

    # Parse input parameters
    draw_markers = draw_markers if draw_markers is not None else len(images) == 1

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
                raise(ValueError('No rows matched for column {}'.format(k)))
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
                raise(ValueError('No rows matched for column {}'.format(k)))
            df_all.append(df)
    else:
        raise ValueError('Unsupported plot definition!')

    if len(images) == 0:
        images = df['image_id'].unique().tolist()

    # Define a parametric model for the trade-off curve
    if metric in {'ssim', 'msssim'}:
        fit_bounds = ([0.1, 1e-5, -1, 0], [3, 10, 7, 0.1])

        def func(x, a, b, c, d):
            return 1/(1 + np.exp(- b * x ** a + c)) - d
    else:
        fit_bounds = ([1e-4, 1e-5, 1e-2, -200], [20, 1000, 10, 200])

        def func(x, a, b, c, d):
            return a * np.log(np.clip(b*x**c + d, a_min=1e-9, a_max=1e9))
            # return a + b * x + c * x ** 2 + d * x **3
            # return a * np.log(b * x ** c + d)

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

    if metric == 'psnr':
        ssim_min = 25
        ssim_max = 45
        metric_label = 'PSNR [dB]'

    elif metric == 'msssim_db' or db_scale:
        ssim_min = 10
        ssim_max = 32
        metric_label = 'MS-SSIM [dB]'

    elif metric == 'ssim':
        ssim_min = 0.85
        ssim_max = 1
        metric_label = 'SSIM'

    elif metric == 'msssim':
        ssim_min = 0.9
        ssim_max = 1
        metric_label = 'MS-SSIM'
    else:
        raise ValueError('Unsupported metric!')

    # Iterate over defined plots and draw data accordingly
    for index, dfc in enumerate(df_all):

        bpps = dfc.loc[dfc['selected'], 'bpp'].values
        ssims = dfc.loc[dfc['selected'], metric].values

        if dump_df:
            print('{} matched {} rows -> {}'.format(labels[index], len(dfc.loc[dfc['selected']]), 'debug-{}.csv'.format(labels[index])))
            dfc.loc[dfc['selected']].to_csv('debug-{}.csv'.format(labels[index]))

        x = np.linspace(bpps.min(), bpps.max(), 256)

        if plot == 'fit':
            # Fit all of the data to a single curve

            popt, pcov = curve_fit(func, bpps, ssims, bounds=fit_bounds, maxfev=10000)
            axes.plot(x, func(x, *popt), styles[index][0], label=labels[index] if add_legend else None)
            # print(labels[index], *popt)

            # If plotting many images, add confidence intervals
            if len(images) > 5 or len(images) == 0:
                a, b, c, d = unc.correlated_values(popt, pcov)
                if metric in {'ssim', 'msssim'}:
                    py = 1 / (1 + unp.exp(- b * x ** a + c)) - d
                else:
                    py = a * unp.log(np.clip(b * x ** c + d, a_min=1e-9, a_max=1e9))

                nom = unp.nominal_values(py)
                std = unp.std_devs(py)

                axes.plot(x, nom - 1.96 * std, c=styles[index][0][0], alpha=0.2)
                axes.plot(x, nom + 1.96 * std, c=styles[index][0][0], alpha=0.2)
                axes.fill(np.concatenate([x, x[::-1]]), np.concatenate([nom - 1.96 * std, (nom + 1.96 * std)[::-1]]),
                          alpha=0.1, fc=styles[index][0][0], ec='None')

            ssim_min = min([ssim_min, func(x[0], *popt)]) if update_ylim else ssim_min

        elif plot == 'ensemble':
            # Fit individual images to a curve, then average the curves

            Y = np.zeros((len(images), len(x)))

            for image_no, image_id in enumerate(images):

                bpps = dfc.loc[dfc['selected'] & (dfc['image_id'] == image_id), 'bpp'].values
                ssims = dfc.loc[dfc['selected'] & (dfc['image_id'] == image_id), metric].values

                try:
                    popt, pcov = curve_fit(func, bpps, ssims,
                                       bounds=fit_bounds,
                                       maxfev=10000)
                    ssims_est = func(bpps, *popt)
                    mse = np.sum(np.power(ssims - ssims_est, 2))
                    print('MSE for {}:{} = {:.2f}'.format(labels[index], image_no, mse))
                    if mse > 10:
                        print('WARNING Large MSE for {}:{} = {:.2f}'.format(labels[index], image_no, mse))
                        print('  bounds: ', fit_bounds)
                        print('  params: ', popt)

                except RuntimeError:
                    print('ERROR', labels[index], 'image =', image_id, 'bpp =', bpps, 'ssims =', ssims)

                Y[image_no] = func(x, *popt)
                # out_of_range_mask = (x < 0.33 * np.min(bpps)) + (x > np.max(bpps) * 3)
                # Y[image_no, out_of_range_mask] = np.nan

            # print(Y.tolist())
            y = np.nanmean(Y, axis=0)
            if db_scale:
                y = -10*np.log10(1 - y)
            axes.plot(x, y, styles[index][0], label=labels[index] if add_legend else None)
            ssim_min = min([ssim_min, min(y)]) if update_ylim else ssim_min

        elif plot == 'line':
            # Simple linear interpolation

            axes.plot(bpps, ssims, styles[index][0], label=labels[index] if add_legend else None)
            ssim_min = min([ssim_min, min(ssims)]) if update_ylim else ssim_min

        elif plot == 'averages':
            # For each quality level (QF, #channels) find the average quality level

            dfa = dfc.loc[dfc['selected']]

            if 'n_features' in dfa:
                dfg = dfa.groupby('n_features')
            else:
                dfg = dfa.groupby('quality')

            bpps = dfg.mean()['bpp'].values
            ssims = dfg.mean()[metric].values

            axes.plot(bpps, ssims, styles[index][0], label=labels[index] if add_legend else None, marker=avg_markers[index], alpha=0.65)
            ssim_min = min([ssim_min, min(ssims)]) if update_ylim else ssim_min

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

                if db_scale:
                    dfc[metric] = -10 * np.log10(1 - dfc[metric])

                g = sns.scatterplot(data=dfc[dfc['selected']], x='bpp', y=metric,
                                palette="Set2", ax=axes, legend=detailed_legend,
                                **style_mapping)

            else:
                if db_scale:
                    ssims = -10 * np.log10(1 - ssims)
                axes.plot(bpps, ssims, styles[index][1], alpha=10 / (sum(dfc['selected'])))

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
    axes.set_ylim([ssim_min * 0.99, ssim_max])
    # axes.set_ylim([0.75, 1])
    # axes.legend(loc='lower right')
    axes.set_title(title)
    axes.set_xlabel('Effective bpp')
    axes.set_ylabel(metric_label)

