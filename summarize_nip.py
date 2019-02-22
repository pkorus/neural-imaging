from collections import defaultdict
import json
import sys
import os

dirname = './data/raw/nip_model_snapshots/'

cameras = sorted(os.listdir(dirname))

print('Showing stats for {}'.format(dirname))

stats = {'psnr': defaultdict(lambda: []), 'ssim': defaultdict(lambda: [])}

for camera in cameras:
    print('\n', camera)
    pipelines = sorted(os.listdir(os.path.join(dirname, camera)))
    for pipe in pipelines:
        with open(os.path.join(dirname, camera, pipe, 'progress.json')) as f:
            ts = json.load(f)
        print('  {:10s} -> PSNR: {:.2f} dB\tSSIM: {:.3f}'.format(pipe, ts['psnr'][-1], ts['ssim'][-1]))
        for k, v in stats.items():
            v[pipe].append(ts[k][-1])

for k, v in stats.items():
    print('\n# Mean {}'.format(k))
    for pipe in v.keys():
        print('{:10s}: {:.3f}'.format(pipe, sum(v[pipe]) / len(v[pipe])))

    # print(v)
