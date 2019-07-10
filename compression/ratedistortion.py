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
from scipy.optimize import curve_fit

from helpers import loading, utils
from compression import jpeg_helpers, afi


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
        df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'bytes', 'bpp'])

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

                    df = df.append({'image_id': image_id,
                                    'filename': filename,
                                    'codec': 'jpeg',
                                    'quality': q,
                                    'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                    'psnr': compare_psnr(image, image_compressed, data_range=1),
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
            df = pd.DataFrame(columns=['image_id', 'filename', 'codec', 'quality', 'ssim', 'psnr', 'bytes', 'bpp'])

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

                        df = df.append({'image_id': image_id,
                                        'filename': filename,
                                        'codec': 'jpeg2000',
                                        'quality': q,
                                        'ssim': compare_ssim(image, image_compressed, multichannel=True, data_range=1),
                                        'psnr': compare_psnr(image, image_compressed, data_range=1),
                                        'bytes': image_bytes,
                                        'bpp': 8 * image_bytes / image.shape[0] / image.shape[1]
                                        }, ignore_index=True)

                        pbar.set_postfix(image_id=image_id, quality=q)
                        pbar.update(1)

            df.to_csv(os.path.join(directory, 'jpeg2000.csv'), index=False)

        return df


def get_dcn_df(directory, model_directory, write_files=False, force_calc=False):

    # Discover test files
    files, _ = loading.discover_files(directory, n_images=-1, v_images=0)
    batch_x = loading.load_images(files, directory, load='y')
    batch_x = batch_x['y'].astype(np.float32) / (2 ** 8 - 1)

    # Create a new table for the DCN
    df = pd.DataFrame(
        columns=['image_id', 'filename', 'codec', 'ssim', 'psnr', 'entropy', 'bytes', 'bpp', 'layers', 'quantization',
                 'entropy_reg', 'codebook', 'latent', 'latent_shape', 'n_features'])

    # Discover available models
    model_dirs = list(Path(model_directory).glob('**/progress.json'))
    print('Found {} models'.format(len(model_dirs)))

    df_path = os.path.join(directory, 'dcn.csv')

    if os.path.isfile(df_path) and not force_calc:
        print('Restoring DCN stats from {}'.format(df_path))
        df = pd.read_csv(df_path, index_col=False)
    else:

        for model_dir in model_dirs:
            print('Processing: {}'.format(model_dir))
            dcn, stats = afi.restore_model(os.path.split(str(model_dir))[0], batch_x.shape[1], fetch_stats=True)

            # Dump compressed images
            for image_id, filename in enumerate(files):

                try:
                    batch_y, image_bytes = afi.dcn_simulate_compression(dcn, batch_x[image_id:image_id + 1])
                except Exception as e:
                    print('Error while processing {} with {} : {}'.format(filename, dcn.model_code, e))
                    raise e

                # Save the image
                image_dir = os.path.join(directory, os.path.splitext(filename)[0])
                if not os.path.isdir(image_dir):
                    os.makedirs(image_dir)

                if write_files:
                    image_path = os.path.join(image_dir, dcn.model_code.replace('/', '-') + '.png')
                    imageio.imwrite(image_path, (255 * batch_y[0]).astype(np.uint8))

                df = df.append({'image_id': image_id,
                                'filename': filename,
                                'codec': dcn.model_code,
                                'ssim': compare_ssim(batch_x[image_id], batch_y[0], multichannel=True, data_range=1),
                                'psnr': compare_psnr(batch_x[image_id], batch_y[0], data_range=1),
                                'entropy': stats['entropy'],
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


def plot_curve(plots, axes, dirname='../data/clic256', images=[], use_kde=False, draw_markers=None, metric='ssim', title=None):
    draw_markers = draw_markers if draw_markers is not None else len(images) == 1

    df_all = []

    def func(x, a, b, c, d):
        return d - a * np.exp(-b * x ** c)

    for filename, selectors in plots:
        df = pd.read_csv(os.path.join(dirname, filename), index_col=False)
        for k, v in selectors.items():
            df = df[df[k] == v]
        df_all.append(df)

    # Select measurements for the given images
    for dfc in df_all:
        if len(images) > 0:
            dfc['selected'] = dfc['image_id'].apply(lambda x: x in images)
        else:
            dfc['selected'] = True

    labels = ['dcn', 'jpeg', 'jpeg2k']
    styles = [['g-', 'gx'], ['r-', 'rx'], ['b-', 'b+']]

    ssim_min = 1
    for index, dfc in enumerate(df_all):

        bpps = dfc.loc[dfc['selected'], 'bpp'].values
        ssims = dfc.loc[dfc['selected'], metric].values

        # Discard invalid measurements (e.g., for empty images)
        bpps = bpps[ssims > 0.5]
        ssims = ssims[ssims > 0.5]

        target_bpps = np.linspace(bpps.min(), bpps.max(), 100)

        if use_kde:
            x, y = utils.ma_gaussian(bpps, ssims, 0.05, 0.2)
            axes.plot(x, y, styles[index][0], label=labels[index])
        else:
            popt, pcov = curve_fit(func, bpps, ssims, bounds=([0, 1e-3, 1e-3, 0.9], [1e6, 20, 5, 1]), maxfev=1000)
            # print(labels[index], *popt)
            axes.plot(target_bpps, func(target_bpps, *popt), styles[index][0], label=labels[index])

        # plt.plot(bpps, ssims, styles[index][1])

        if draw_markers:
            if 'entropy_reg' in dfc:
                # pass
                sns.scatterplot(data=dfc[dfc['selected']], x='bpp', y=metric,
                                hue='n_features',
                                size='entropy_reg',
                                # hue='',
                                palette="Set2",
                                ax=axes, legend='full')
            else:
                axes.plot(bpps, ssims, styles[index][1], alpha=5 / (sum(dfc['selected'])))

        ssim_min = min([ssim_min, func(target_bpps[0], *popt)])

    n_images = len(dfc.loc[dfc['selected'], 'image_id'].unique())

    title = '{} : {}'.format(
        title if title is not None else os.path.split(dirname)[-1],
        '{} images'.format(n_images) if n_images > 1 else dfc.loc[dfc['selected'], 'filename'].unique()[0]
    )

    axes.set_xlim([0, 4])
    axes.set_ylim([ssim_min * 0.99, 1])
    # axes.set_ylim([0.75, 1])
    axes.legend(loc='lower right')
    axes.set_title(title)
    axes.set_xlabel('Effective bpp')
    axes.set_ylabel('SSIM')
