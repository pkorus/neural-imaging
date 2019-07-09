import io
import os
import tqdm
import imageio
import pandas as pd
import numpy as np
import PIL
from skimage.measure import compare_ssim, compare_psnr

from helpers import loading
from compression import jpeg_helpers

import glymur

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
        quality_levels = np.arange(20, 45, 1)
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
                        # image_bytes = os.path.getsize('/tmp/image.jp2')
                        image_bytes = jpeg_helpers.jp2bytes('/tmp/image.jp2')
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