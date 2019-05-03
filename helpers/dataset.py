import numpy as np
from helpers import loading

class IPDataset:
    
    def __init__(self, data_directory, randomize=False, load='xy', val_patch_size=128, val_n_patches=2, n_images=120, v_images=30):
        self.files = {}
        self.files['training'], self.files['validation'] = loading.discover_files(data_directory, randomize=randomize, n_images=n_images, v_images=v_images)
                
        self.data = {
            'training': loading.load_images(self.files['training'], data_directory=data_directory, load=load),
            'validation': loading.load_patches(self.files['validation'], data_directory=data_directory, patch_size=val_patch_size // 2, n_patches=val_n_patches, load=load, discard_flat=True)
        }
        
        if 'x' in self.data['training']:
            self.H, self.W = self.data['training']['x'].shape[1:3]
        else:
            self.H, self.W = self.data['training']['y'].shape[1:3]
            
        print('Loaded dataset with {}x{} images'.format(self.W, self.H))
        
    def __getitem__(self, key):
        if key in ['training', 'validation']:
            return self.data[key]
        else:
            return super().__getitem__(key)
        
    def next_training_batch(self, batch_id, batch_size, patch_size, discard_flat=True):
        batch_x = np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32)
        for b in range(batch_size):
            
            found = False            
            panic_counter = 5
            
            while not found:
                xx = np.random.randint(0, self.W - patch_size)
                yy = np.random.randint(0, self.H - patch_size)

                patch = self.data['training']['y'][batch_id * batch_size + b, yy:yy + patch_size, xx:xx + patch_size, :].astype(np.float) / (2**8 - 1)

                # Check if the found patch is acceptable: eliminate empty patches
                if discard_flat:
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
            
            batch_x[b, :, :, :] = patch
        return batch_x

    def next_validation_batch(self, batch_id, batch_size, output_patch_size=None):
        patch_size = self.data['validation']['y'].shape[1]
        batch_x = np.zeros((batch_size, patch_size, patch_size, 3), dtype=np.float32)
        for b in range(batch_size):
            batch_x[b, :, :, :] = self.data['validation']['y'][batch_id * batch_size + b].astype(np.float)
        if output_patch_size is None or output_patch_size == patch_size:
            return batch_x
        else:
            return batch_x[:, :output_patch_size, :output_patch_size, :]