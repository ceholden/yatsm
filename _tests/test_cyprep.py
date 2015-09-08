import unittest

import numpy as np

import yatsm._cyprep


class TestCyPrep(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Test data
        n_band = 7
        n_mask = 50
        n_images = 1000

        cls.data = np.random.randint(
            0, 10000, size=(n_band, n_images)).astype(np.int32)
        for b in range(n_band):
            cls.data[b, np.random.choice(np.arange(0, n_images),
                                         size=n_mask, replace=False)] = 16000

        cls.mins = np.repeat(0, n_band).astype(np.int16)
        cls.maxes = np.repeat(10000, n_band).astype(np.int16)

    def test_get_valid_mask(self):
        truth = np.all([((b > _min) & (b < _max)) for b, _min, _max in
                        zip(np.rollaxis(self.data, 0),
                            self.mins,
                            self.maxes)], axis=0)

        np.testing.assert_equal(
            truth,
            yatsm._cyprep.get_valid_mask(self.data, self.mins, self.maxes))


if __name__ == '__main__':
    unittest.main()
