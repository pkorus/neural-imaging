import os, sys

sys.path.append('..')
sys.path.append('.')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def usage():
    print('python3 tf_list_nip_variables.py <NIP Model>')

def main(argv):

    if len(argv) != 1:
        usage()
        sys.exit(0)

    if argv[0] not in ['UNet', 'DNet', 'CANNet', 'INet']:
        print('ERROR Unsupported model!')
        sys.exit(1)

    from models import pipelines

    nip_model = getattr(pipelines, argv[0])
    model = nip_model()
    print(model.summary())

    for i, p in enumerate(sorted(model.parameters, key=lambda x: x.name)):
        print('  {1:3d}. {0.name:30s} -> {0.shape}'.format(p, i))

if __name__ == '__main__':
    main(sys.argv[1:])
