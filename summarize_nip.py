from collections import defaultdict
import json
import os
import sys
import argparse


def mean(x):
    return sum(x) / len(x)


def summarize_results(dirname, show_stats=True, n=1):

    cameras = sorted(os.listdir(dirname))

    print('\n# {}'.format(dirname))

    stats = {'psnr': defaultdict(lambda: []), 'ssim': defaultdict(lambda: [])}

    for camera in cameras:
        print('\n  {}'.format(camera))
        pipelines = sorted(os.listdir(os.path.join(dirname, camera)))

        for pipe in pipelines:
            with open(os.path.join(dirname, camera, pipe, 'progress.json')) as f:
                ts = json.load(f)
            data = ts if 'psnr' in ts else ts['Performance']

            print('    {:10s} -> PSNR: {:.2f} dB\tSSIM: {:.3f}'.format(pipe, mean(data['psnr'][-n:]), mean(data['ssim'][-n:])))

            for k, v in stats.items():
                v[pipe].append(mean(data[k][-n:]))

    if show_stats:
        for k, v in stats.items():
            print('\n  ! Mean {}'.format(k))
            for pipe in v.keys():
                print('    {:10s}: {:.3f}'.format(pipe, sum(v[pipe]) / len(v[pipe])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize results of NIP training')
    parser.add_argument('dirs', nargs='*', default=['./data/raw/nip_model_snapshots/'])
    parser.add_argument('--stats', dest='stats', action='store_true', default=False,
                        help='Display summary stats')
    parser.add_argument('--n', dest='n', action='store', default=1, type=int,
                        help='Set > 1 to average last N samples')

    args = parser.parse_args()

    for dirname in args.dirs:
        if os.path.exists(dirname):
            summarize_results(dirname, args.stats, args.n)
        else:
            print('Error: directory {} does not exist!'.format(os.path.abspath(dirname)))
            sys.exit(1)
