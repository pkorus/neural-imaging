#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse
import subprocess
from helpers import coreutils

OK_STR = '\033[92m ok \033[00m'
MISS_STR = '\033[91m missing \033[00m'
FAIL_STR = '\033[91m failed \033[00m'

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
        print('    {:84s} [{}]'.format(filename.format(args.camera), OK_STR if status else MISS_STR))

    # Check performance
    training_log = os.path.join(args.root_dir, config['performance']['file'].format(args.camera))

    if not os.path.isfile(training_log):
        print('ERROR file {} does not exist!'.format(training_log))
        sys.exit(1)

    print('\n  Checking obtained performance:')
    with open(training_log) as f:
        perf = json.load(f)
        for key, expected_value in config['performance']['values'].items():
            obtained_value = coreutils.getkey(perf, key)[-1]
            print('    {:70s} {:5.2f} > {:5.2f} [{}]'.format(key, obtained_value, expected_value, OK_STR if obtained_value > expected_value else FAIL_STR))


def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera', default='D90')
    parser.add_argument('--dir', dest='root_dir', action='store', default='/tmp/neural-imaging-framework',
                        help='output directory for temporary results, default: /tmp/neural-imaging-framework')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='print the output of tested tools, default: false')
    parser.add_argument('--keep', dest='keep', action='store_true', default=False,
                        help='do not remove the test root directory')
    parser.add_argument('--tests', dest='tests', action='store', default=None,
                        help='list of tests to run')

    args = parser.parse_args()

    with open('tests/framework.json') as f:
        settings = json.load(f)

    if os.path.exists(args.root_dir) and not args.keep:
        print('> deleting {}'.format(args.root_dir))
        shutil.rmtree(args.root_dir)

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    if args.tests is None:
        tests = ['train-nip', 'resume-nip', 'train-manipulation', 'train-dcn', 'train-manipulation-dcn']
    else:
        tests = args.tests.split(',')

    for test in tests:
        run_test(settings[test], args)


if __name__ == "__main__":
    main()
