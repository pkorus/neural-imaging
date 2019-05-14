import unittest
from helpers import paramspec


class TestParamSpec(unittest.TestCase):

    def setUp(self):
        # Define an example parameter specification
        self.p = paramspec.ParamSpec({
                    'n_filters': (8, int, (2, None)),
                    'kernel': (5, int, {3, 5, 7, 9, 11}),
                    'n_layers': (3, int, (1, 20)),
                    'dropout': (True, bool, None),
                    'rounding': ('soft', str, {'identity', 'soft', 'soft-codebook', 'sin'}),
                    'train_codebook': (False, bool, None),
                    'activation': ('relu', str, lambda x: 'elu' in x)
        })

    def test_access(self):
        self.assertEqual(self.p.n_filters, 8, 'Cannot access n_latent')

    def test_update_in_range(self):
        self.p.update(n_layers=3)
        self.assertEqual(self.p.n_layers, 3, 'Cannot access n_layers')

    def test_update_out_range(self):
        with self.assertRaises(ValueError):
            self.p.update(n_layers=0)

    def test_update_not_in_enum(self):
        with self.assertRaises(ValueError):
            self.p.update(rounding='invalid')

    def test_custom_validator(self):
        with self.assertRaises(ValueError):
            self.p.update(activation='lu')

    def test_existence(self):
        self.assertEqual('n_layers' in self.p, True, 'Cannot test existence!')

    def test_to_dict(self):
        params = self.p.to_dict()
        status = {key in params for key in ['n_filters', 'kernel', 'n_layers', 'dropout', 'rounding', 'train_codebook', 'activation']}
        self.assertEqual(all(status), True, 'Dictionary does not contain all attributes!')


if __name__ == '__main__':
    unittest.main()
