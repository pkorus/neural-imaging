import numpy as np
from helpers import loading


class IPDataset(object):
    
    def __init__(self, data_directory, randomize=False, load='xy', val_rgb_patch_size=128, val_n_patches=2, n_images=120, v_images=30):
        
        if not any(load == allowed for allowed in ['xy', 'x', 'y']):
            raise ValueError('Invalid X/Y data requested!')
        
        self.files = {}
        self._loaded_data = load
        self.files['training'], self.files['validation'] = loading.discover_files(data_directory, randomize=randomize, n_images=n_images, v_images=v_images)
                
        self.data = {
            'training': loading.load_images(self.files['training'], data_directory=data_directory, load=load),
            'validation': loading.load_patches(self.files['validation'], data_directory=data_directory, patch_size=val_rgb_patch_size // 2, n_patches=val_n_patches, load=load, discard_flat=True)
        }
        
        if 'y' in self.data['training']:
            self.H, self.W = self.data['training']['y'].shape[1:3]
        else:
            self.H, self.W = (2 * dim for dim in self.data['training']['x'].shape[1:3])
            
        print('Loaded dataset with {}x{} images'.format(self.W, self.H))
        
    def __getitem__(self, key):
        if key in ['training', 'validation']:
            return self.data[key]
        else:
            raise KeyError('Key: {} not found!'.format(key))
        
    def next_training_batch(self, batch_id, batch_size, rgb_patch_size, discard_flat=False):
        
        if discard_flat and 'y' not in self.data['training']:
            raise ValueError('Cannot discard flat patches if RGB data is not loaded.')
        
        if (batch_id + 1) * batch_size > len(self.files['training']):
            raise ValueError('Not enough images for the requested batch_id & batch_size')
        
        raw_patch_size = rgb_patch_size // 2
        
        # Allocate memory for the batch
        batch = {
            'x': np.zeros((batch_size, raw_patch_size, raw_patch_size, 4), dtype=np.float32) if 'x' in self._loaded_data else None,
            'y': np.zeros((batch_size, rgb_patch_size, rgb_patch_size, 3), dtype=np.float32) if 'y' in self._loaded_data else None
        }
        
        for b in range(batch_size):
            
            found = False            
            panic_counter = 5
            
            while not found:
                xx = np.random.randint(0, self.W - rgb_patch_size)
                yy = np.random.randint(0, self.H - rgb_patch_size)

                # Check if the found patch is acceptable: eliminate empty patches
                if discard_flat:
                    patch = self.data['training']['y'][batch_id * batch_size + b, yy:yy + rgb_patch_size, xx:xx + rgb_patch_size, :].astype(np.float) / (2**8 - 1)
                  
                    patch_variance = np.var(patch)
                    if patch_variance < 1e-2:
                        panic_counter -= 1
                        found = False if panic_counter > 0 else True
                    elif patch_variance < 0.02:
                        found = np.random.uniform() > 0.5
                    else:
                        found = True
                else:
                    found = True
            
            if 'x' in self._loaded_data:
                batch['x'][b, :, :, :] = self.data['training']['x'][batch_id * batch_size + b , yy//2:yy//2 + raw_patch_size, xx//2:xx//2 + raw_patch_size, :].astype(np.float) / (2**16 - 1)
            if 'y' in self._loaded_data:
                batch['y'][b, :, :, :] = self.data['training']['y'][batch_id * batch_size + b, yy:yy + rgb_patch_size, xx:xx + rgb_patch_size, :].astype(np.float) / (2**8 - 1)

        if self._loaded_data == 'xy':
            return batch['x'], batch['y']
        elif self._loaded_data == 'y':
            return batch['y']
        elif self._loaded_data == 'x':
            return batch['x']

    def next_validation_batch(self, batch_id, batch_size):
        
        # RGB patch size
        if 'y' in self._loaded_data:
            patch_size = self.data['validation']['y'].shape[1]
        else:
            patch_size = 2 * self.data['validation']['x'].shape[1]
            
        batch = {
            'x': np.zeros((batch_size, patch_size // 2, patch_size // 2, 4), dtype=np.float32) if 'x' in self._loaded_data else None,
            'y': np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32) if 'y' in self._loaded_data else None
        }
        
        for b in range(batch_size):
            if 'x' in self._loaded_data:
                batch['x'][b, :, :, :] = self.data['validation']['x'][batch_id * batch_size + b].astype(np.float) / (2**16 - 1)
            if 'y' in self._loaded_data:
                batch['y'][b, :, :, :] = self.data['validation']['y'][batch_id * batch_size + b].astype(np.float) / (2**8 - 1)
        
        if self._loaded_data == 'xy':
            return batch['x'], batch['y']
        elif self._loaded_data == 'y':
            return batch['y']
        elif self._loaded_data == 'x':
            return batch['x']
    
    @property
    def count_training(self):
        key = self._loaded_data[0]
        return self.data['training'][key].shape[0]
    
    @property
    def count_validation(self):
        key = self._loaded_data[0]
        return self.data['validation'][key].shape[0]
