""" Tests for yatsm.phenology.longtermmean
"""
import numpy as np
import pytest

import yatsm.phenology.longtermmean as ltm


def test_scale_EVI(data):
    evi_norm = ltm.scale_EVI(data['evi'].values, data['period'].values)
    np.testing.assert_almost_equal(evi_norm.max(), 1.69351, decimal=5,
                                   err_msg='Scaled EVI max is not correct')
    np.testing.assert_almost_equal(evi_norm.min(), -0.2385107, decimal=5,
                                   err_msg='Scaled EVI min is not correct')


@pytest.mark.parametrize(('arr', 'ans'), [
    (np.arange(10), 4),
    (np.arange(-10, 10), 10),
    (np.linspace(0, 1, 10), 4),
    (np.linspace(0, 1, 20), 10),
    (np.linspace(-1, 1, 25), 12)
])
def test_halfmax(arr, ans):
    test = ltm.halfmax(arr)
    assert test == ans


def test_pheno_fit(data):
    result = ltm.longtermmeanphenology(data['evi'], periods=data['period'],
                                       year_interval=3, q_min=10., q_max=90.)
    # Test
    np.testing.assert_almost_equal(result.corrcoef, 0.9595954, decimal=5,
                                   err_msg='Spline correlation is not correct')
    np.testing.assert_equal(result.springDOY, 138,
                            err_msg='Spring LTM is not correct')
    np.testing.assert_equal(result.autumnDOY, 283,
                            err_msg='Autumn LTM is not correct')
