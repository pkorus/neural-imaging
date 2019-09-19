import os
import sys
import argparse

from helpers.results_data import nip_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize results of NIP training')
    parser.add_argument('dirs', nargs='*', default=['./data/models/nip'])
    parser.add_argument('--stats', dest='stats', action='store_true', default=False,
                        help='Display summary stats')
    parser.add_argument('--n', dest='n', action='store', default=1, type=int,
                        help='Set > 1 to average last N samples')

    args = parser.parse_args()

    for dirname in args.dirs:
        if os.path.exists(dirname):
            print('\n# {}'.format(dirname))
            df = nip_stats(dirname, args.n)
            print(df.to_string())

            if args.stats:
                print('Per-pipeline summary:')
                print(df.groupby('pipeline').mean().reset_index().to_string())

        else:
            print('Error: directory {} does not exist!'.format(os.path.abspath(dirname)))
            sys.exit(1)
