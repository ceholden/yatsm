""" Test for `yatsm.regression.transforms`
"""
import numpy as np
import patsy
import py.test

from yatsm.regression.transforms import harm


def test_harmonic_transform():
    x = np.arange(735688, 735688 + 100, 1)
    design = patsy.dmatrix('0 + harm(x, 1)')

    truth = np.vstack((np.cos(2 * np.pi / 365.25 * x),
                       np.sin(2 * np.pi / 365.25 * x))).T

    np.testing.assert_equal(np.asarray(design), truth)
