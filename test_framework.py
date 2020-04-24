#!/usr/bin/env python3
import os
import sys
import json
import shutil
import argparse

from helpers import utils, tf_helpers

OK_STR = '\033[92m ok \033[00m'
MISS_STR = '\033[91m missing \033[00m'
FAIL_STR = '\033[91m failed \033[00m'


def run_test(test_name, config, args):

    if not args.verbose:
        log_path = os.path.join(args.root_dir, test_name)
    else:
        log_path = None

    code = utils.shell(config['command'].format(cam=args.camera, root=args.root_dir), log_path, verbosity=1)
    print('\n  Exit code: {}\n'.format(code))

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
            obtained_value = utils.get(perf, key, sep='/')[-1]
            print('    {:70s} {:5.2f} > {:5.2f} [{}]'.format(key, obtained_value, expected_value, OK_STR if obtained_value > expected_value else FAIL_STR))


def main():
    parser = argparse.ArgumentParser(description='Train a neural imaging pipeline')
    parser.add_argument('--cam', dest='camera', action='store', help='camera', default='D90')
    parser.add_argument('--dir', dest='root_dir', action='store', default='/tmp/neural-imaging',
                        help='output directory for temporary results, default: /tmp/neural-imaging')
    parser.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='print the output of tested tools, default: false')
    parser.add_argument('--keep', dest='keep', action='store_true', default=False,
                        help='do not remove the test root directory')
    parser.add_argument('--tests', dest='tests', action='store', default=None,
                        help='list of tests to run')

    args = parser.parse_args()

    utils.setup_logging()
    tf_helpers.disable_warnings()
    tf_helpers.print_versions()

    with open('config/tests/framework.json') as f:
        settings = json.load(f)

    if os.path.exists(args.root_dir) and not args.keep:
        print('\n> deleting {}'.format(args.root_dir))
        shutil.rmtree(args.root_dir)

    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    if args.tests is None:
        tests = ['train-nip', 'resume-nip', 'train-manipulation', 'train-dcn', 'train-manipulation-dcn']
    else:
        tests = args.tests.split(',')

    for test in tests:
        run_test(test, settings[test], args)


if __name__ == "__main__":
    main()
