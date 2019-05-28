import unittest
from compression import jpeg_helpers


class JpegTest(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_stats(self):
        jpeg = jpeg_helpers.JPEGStats('../data/samples/watkins.jpg')

        # Test selected markers
        self.assertIn('SOI', jpeg.blocks)
        self.assertIn('EOI', jpeg.blocks)
        self.assertIn('DHT:0', jpeg.blocks)
        self.assertIn('ECD', jpeg.blocks)

        # Test image measurements
        self.assertEqual(jpeg.get_bytes(), 107396)
        self.assertEqual(jpeg.get_effective_bytes(), 101279)
        self.assertEqual(jpeg.image.shape, (425, 729, 3))
        self.assertEqual(round(jpeg.get_bpp(), 2), 2.77)
        self.assertEqual(round(jpeg.get_effective_bpp(), 2), 2.62)
