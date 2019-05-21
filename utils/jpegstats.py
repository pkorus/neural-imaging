import sys
sys.path.append('.')
from helpers import jpeg_helpers

# Parse the file
jpeg = jpeg_helpers.JPEGStats(sys.argv[-1])

print('Input file       : {}'.format(sys.argv[-1]))
print('File size        : {:,} bytes'.format(jpeg.get_bytes()))
print('Image data size  : {:,} bytes'.format(jpeg.get_effective_bytes()))
print('Image size       : {0}'.format(jpeg.image.shape))
print('Rate             : {:.2f} bpp'.format(jpeg.get_bpp()))
print('Rate (effective) : {:.2f} bpp'.format(jpeg.get_effective_bpp()))


# Print stats
print('Markers:\n')
print(' Address                 Marker Description')
print(' -------                 ------ -----------')
for key, value in jpeg.blocks.items():
    try:
        description = jpeg_helpers.markers[key.split(':')[0]]
    except KeyError:
        description = 'n/a'

    print(' 0x{0:08x} {0:8,d} {1:>10s} ({2:s})'.format(value, key, description))
