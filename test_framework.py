import os
import sys
import json
import shutil
import argparse
from functools import reduce
import subprocess


def shell(command, log=None):

    print('\n> {}\n'.format(command))

    if log is None:
        p = subprocess.Popen(command, shell=True)
    else:
        print('  log: {}'.format(log))
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    p.wait()

    # Dump stdout
    if log is not None:
        with open(log, 'w') as f:
            f.write('# STDOUT\n')
            f.write(p.stdout.read().decode('utf-8'))
            f.write('# STDERR\n')
            f.write(p.stderr.read().decode('utf-8').replace('\r', '\n'))

    return p.returncode


def run_test(config, args):

    code = shell(config['command'].format(cam=args.camera, root=args.root_dir), os.path.join(args.root_dir, config['log']) if not args.verbose else None)
    print('  exit code: {}\n'.format(code))

    if code != 0:
        print('ERROR non-zero return code for {}'.format('nip-training'))
        sys.exit(1)

    # Check the output files
    print('  Checking expected files:')
    for filename in config['files']:
        status = os.path.isfile(os.path.join(args.root_dir, filename.format(args.camera)))
        print('    {:70s} [{}]'.format(filename.format(args.camera), 'ok' if status else 'missing'))
        if not status:
            print('ERROR file {} does not exist!'.format(filename.format(args.camera)))
            sys.exit(1)

    # Check performance
    print('\n  Checking obtained performance:')
    with open(os.path.join(args.root_dir, config['performance']['file'].format(args.camera))) as f:
        perf = json.load(f)
        for key, expected_value in config['performance']['values'].items():
            obtained_value = reduce(lambda c, k: c.get(k, {}), key.split('/'), perf)[-1]
            print('    {:56s} {:5.2f} > {:5.2f} [{}]'.format(key, obtained_value, expected_value, 'ok' if obtained_value > expected_value else 'failed'))


def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera', default='Nikon D90')
    parser.add_argument('--dir', dest='root_dir', action='store', default='/tmp/neural-imaging-framework',
                        help='output directory for temporary results, default: /tmp/neural-imaging-framework')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='print the output of tested tools, default: false')

    args = parser.parse_args()

    with open('tests/framework.json') as f:
        settings = json.load(f)

    if os.path.exists(args.root_dir):
        print('> deleting {}'.format(args.root_dir))
        shutil.rmtree(args.root_dir)

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    run_test(settings['train-nip'], args)
    run_test(settings['resume-nip'], args)
    run_test(settings['train-manipulation'], args)


if __name__ == "__main__":
    main()
