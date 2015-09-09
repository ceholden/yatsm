import numpy as np
import pytest
import yatsm._cyprep as cyprep


def test_get_valid_mask():
    n_bands, n_images, n_mask = 8, 500, 50
    data = np.random.randint(0, 10000,
                             size=(n_bands, n_images)).astype(np.int32)
    # Add in bad data
    _idx = np.arange(0, n_images)
    for b in range(n_bands):
        idx = np.random.choice(_idx, size=n_mask, replace=False)
        data[b, idx] = 16000

    mins = np.repeat(0, n_bands).astype(np.int16)
    maxes = np.repeat(10000, n_bands).astype(np.int16)

    truth = np.all([((b > _min) & (b < _max)) for b, _min, _max in
                    zip(np.rollaxis(data, 0), mins, maxes)], axis=0)

    np.testing.assert_equal(truth, cyprep.get_valid_mask(data, mins, maxes))
