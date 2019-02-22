import os
import numpy as np
import tqdm
import imageio
from helpers import coreutils


def discover_files(data_directory, n_images=120, v_images=30, extension='png', randomize=False):
    """
    Find available images and split them into training / validation sets.
    :param data_directory: directory
    :param n_images: number of training images
    :param v_images: number of validation images
    :param extension: file extension
    :param randomize: whether to shuffle files before the split
    """

    files = coreutils.listdir(data_directory, '.*\.{}$'.format(extension))
    print('In total {} files available'.format(len(files)))

    if randomize:
        np.random.shuffle(files)

    if len(files) >= n_images + v_images:
        val_files = files[n_images:(n_images + v_images)]
        files = files[0:n_images]
    else:
        raise ValueError('Not enough images!')
        
    return files, val_files


def load_fullres(files, data_directory, extension='png'):
    """
    Load pairs of full-resolution images: (raw, rgb). Raw inputs are stored in *.npy files (see
    train_prepare_training_set.py).
    :param files: list of files to be loaded
    :param data_directory: directory path
    :param extension: file extension of rgb images
    """
    n_images = len(files)
    
    # Check image resolution
    image = imageio.imread(os.path.join(data_directory, files[0]))
    resolutions = (image.shape[0] >> 1, image.shape[1] >> 1)
    del image
    
    data_x = np.zeros((n_images, *resolutions, 4), dtype=np.uint16)
    data_y = np.zeros((n_images, 2 * data_x.shape[1], 2 * data_x.shape[2], 3), dtype=np.uint8)

    with tqdm.tqdm(total=n_images, ncols=100, desc='Loading data') as pbar:

        for i, file in enumerate(files):
            npy_file = file.replace('.{}'.format(extension), '.npy')
            data_x[i, :, :, :] = np.load(os.path.join(data_directory, npy_file))
            data_y[i, :, :, :] = imageio.imread(os.path.join(data_directory, file))
            pbar.update(1)

        return data_x, data_y

    
def load_patches(files, data_directory, patch_size=128, n_patches=100, discard_flat=False, extension='png'):
    """
    Sample (raw, rgb) pairs or random patches from given images.
    :param files: list of available images
    :param data_directory: directory path
    :param patch_size: patch size (in the raw image - rgb patches will be twice as big)
    :param n_patches: number of patches per image
    :param discard_flat: remove flat patches
    :param extension: file extension of rgb images
    """
    v_images = len(files)
    valid_x = np.zeros((v_images * n_patches, patch_size, patch_size, 4), dtype=np.float32)
    valid_y = np.zeros((v_images * n_patches, 2 * valid_x.shape[1], 2 * valid_x.shape[2], 3), dtype=np.float32)

    with tqdm.tqdm(total=v_images * n_patches, ncols=100, desc='Loading data') as pbar:

        vpatch_id = 0

        for i, file in enumerate(files):
            npy_file = file.replace('.{}'.format(extension), '.npy')
            image_x = np.load(os.path.join(data_directory, npy_file))
            image_y = imageio.imread(os.path.join(data_directory, file))

            H, W = image_x.shape[0:2]

            # Sample random patches
            panic_counter = 100 * n_patches

            for b in range(n_patches):
                found = False

                while not found: 
                    xx = np.random.randint(0, W - patch_size)
                    yy = np.random.randint(0, H - patch_size)
                    valid_x[vpatch_id] = image_x[yy:yy + patch_size, xx:xx + patch_size, :].astype(np.float32) / (2**16 - 1)
                    valid_y[vpatch_id] = image_y[(2*yy):2*(yy + patch_size), (2*xx):2*(xx + patch_size), :].astype(np.float32) / (2**8 - 1)

                    # Check if the found patch is acceptable:
                    # - eliminate empty patches
                    if discard_flat:
                        patch_variance = np.var(valid_y[vpatch_id])
                        if patch_variance < 1e-2:
                            panic_counter -= 1
                            found = False if panic_counter > 0 else True
                        elif patch_variance < 0.02:
                            found = np.random.uniform() > 0.5
                        else:
                            found = True
                    else:
                        found = True
                        
                vpatch_id += 1    
                pbar.update(1)

        return valid_x, valid_y


def load_patches_rgb(files, data_directory, patch_size=128, n_patches=100, discard_flat=False):
    """
    Sample rgb patches from given images.
    :param files: list of available images
    :param data_directory: directory path
    :param patch_size: patch size (in the raw image - rgb patches will be twice as big)
    :param n_patches: number of patches per image
    :param discard_flat: remove flat patches
    """

    v_images = len(files)
    valid_y = np.zeros((v_images * n_patches, patch_size, patch_size, 3), dtype=np.float32)

    with tqdm.tqdm(total=v_images * n_patches, ncols=100, desc='Loading data') as pbar:

        vpatch_id = 0
        for i, file in enumerate(files):
            image_y = imageio.imread(os.path.join(data_directory, file))

            H, W = image_y.shape[0:2]

            # Sample random patches
            panic_counter = 100 * n_patches 
            for b in range(n_patches):
                found = False

                while not found: 
                    xx = np.random.randint(0, W - patch_size)
                    yy = np.random.randint(0, H - patch_size)
                    valid_y[vpatch_id] = image_y[yy:yy + patch_size, xx:xx + patch_size, :].astype(np.float32) / (2**8 - 1)

                    # Check if the found patch is acceptable:
                    # - eliminate empty patches
                    if discard_flat:
                        patch_variance = np.var(valid_y[vpatch_id])
                        if patch_variance < 1e-2:
                            panic_counter -= 1
                            found = False if panic_counter > 0 else True
                        elif patch_variance < 0.02:
                            found = np.random.uniform() > 0.5
                        else:
                            found = True
                    else:
                        found = True
                        
                vpatch_id += 1    
                pbar.update(1)

        return valid_y
