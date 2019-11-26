#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:19:09 2019

@author: pkorus
"""


# %% Compress an image patch with all models

model_directory = '../data/raw/dcn/forensics/'
models = [str(mp.parent.parent) for mp in list(Path(model_directory).glob('**/progress.json'))]
models = sorted(models)

print(models)

image_id = 3

fig, axes = plotting.sub(6 * len(models), ncols=6)

for model_id, model in enumerate(models):

    dcn = codec.restore_model(model, patch_size=batch_x.shape[1])
    match_jpeg(dcn, batch_x[images[image_id]:images[image_id]+1], axes[model_id*6:(model_id+1)*6])
    
    axes[model_id*6].set_ylabel(os.path.relpath(model, model_directory))

# fig.savefig('fig_compare_dcn_models_image_{}.pdf'.format(images[image_id]), bbox_inches='tight')

# %% Compare outputs of the basic compression with optimized ones

def nm(x):
    x = np.abs(x)
    return (x - x.min()) / (x.max() - x.min())

dcn_model = '8k'
save_debug = False

# Define the distribution channel
models = OrderedDict()
models[0.001]   = '../data/raw/dcn/forensics/{}-0.0010'.format(dcn_model) # 95% accuracy
models[0.005]   = '../data/raw/dcn/forensics/{}-0.0050'.format(dcn_model) # 89% accuracy
models[0.010]   = '../data/raw/dcn/forensics/{}-0.0100'.format(dcn_model) # 85% accuracy
models[0.050]   = '../data/raw/dcn/forensics/{}-0.0500'.format(dcn_model) # 72% accuracy
models[0.100]   = '../data/raw/dcn/forensics/{}-0.1000'.format(dcn_model) # 65% accuracy
models[1.000]   = '../data/raw/dcn/forensics/{}-1.0000'.format(dcn_model) # 62% accuracy
# models[1.001]   = '../data/raw/dcn/forensics/{}-1.0000b'.format(dcn_model) # 62% accuracy
# models[1.002]   = '../data/raw/dcn/forensics/{}-1.0000c'.format(dcn_model) # 62% accuracy
# models[5.000]   = '../data/raw/dcn/forensics/{}-5.0000'.format(dcn_model) # 62% accuracy
# models[1000.0]   = '../data/raw/dcn/forensics/{}-1000.0'.format(dcn_model) # 62% accuracy
models['basic'] = '../data/raw/dcn/forensics/{}-basic'.format(dcn_model)  # 62% accuracy

outputs = OrderedDict()
stats = OrderedDict()

# Worst images for clic: 1, 28, 33, 36
image_id = 1 # 32 # 28 for clic

for model in models.keys():
    dcn = codec.restore_model(models[model], patch_size=batch_x.shape[1])
    outputs[model], stats[model] = codec.dcn_compress_n_stats(dcn, batch_x[image_id:image_id+1])

print('# {}'.format(files[image_id]))
for model in models.keys():
    print('{:>10} : ssim = {:.3f} @ {:.3f} bpp'.format(model, stats[model]['ssim'][0], stats[model]['bpp'][0]))

if save_debug:

    fig = plotting.imsc(list(outputs.values()),
                        ['{} : ssim = {:.3f} @ {:.3f} bpp'.format(x, stats[x]['ssim'][0], stats[x]['bpp'][0]) for x in models.keys()],
                        figwidth=24)
    fig.savefig('debug.pdf', bbox_inches='tight')
    
    # # Dump to file
    for key, value in outputs.items():
        os.makedirs('debug/{}/{}/'.format(dataset.strip('/').split('/')[-1], image_id), exist_ok=True)
        imageio.imwrite('debug/{}/{}/{}_{}.png'.format(dataset.strip('/').split('/')[-1], image_id, dcn_model, key), value.squeeze())
        with open('debug/{}/{}/{}_log.txt'.format(dataset.strip('/').split('/')[-1], image_id, dcn_model), 'w') as f:
            f.write('# {}\n'.format(files[image_id]))
            for model in models.keys():
                f.write('{:>10} : ssim = {:.3f} @ {:.3f} bpp\n'.format(model, stats[model]['ssim'][0], stats[model]['bpp'][0]))

# %%

fig = plotting.imsc([nm(outputs[k] - outputs['basic']) for k in outputs.keys()],
                    ['{}'.format(x) for x in models.keys()], figwidth=24)

fig.savefig('diff.pdf', bbox_inches='tight')

# %% Images

from diff_nip import compare_images_ab_ref, fft_log_norm

use_pretrained_ref = False

if use_pretrained_ref:
    reference = outputs['basic']
else:
    reference = batch_x[image_id:image_id+1]

fig = compare_images_ab_ref(reference, outputs[1.000], outputs[0.001],
                            labels=['Pre-trained DCN' if use_pretrained_ref else 'Original image', '$\lambda=1.000$', '$\lambda=0.001$'])

axes_bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('diff_dcn.pdf', bbox_inches='tight', dpi=np.ceil(batch_x.shape[1] / axes_bbox.height))

# %% Perception Distance (AlexNet)

from models import lpips

distance = lpips.lpips(outputs['basic'], batch_x[image_id:image_id+1])

print(distance)

