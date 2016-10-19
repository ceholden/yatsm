""" Tests for ``yatsm.structural_break._ewma``
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from yatsm.structural_break import _ewma


@pytest.fixture
def x():
    # From ``CRAN::qcc``
    x = np.array([33.75, 33.05, 34, 33.81, 33.46, 34.02, 33.68, 33.27, 33.49,
                  33.20, 33.62, 33.00, 33.54, 33.12, 33.84])
    return x


@pytest.mark.parametrize(('k', 'sd'), [
    (2, 0.4261651),
    (3, 0.3603071),
    (4, 0.3286385),
    (5, 0.3118893),
    (6, 0.3093923)
])
def test__sd_moving_range(x, k, sd):
    result = _ewma._sd_moving_range(x, k)
    assert abs(sd - result) < 1e-5


def test__c4():
    n = np.arange(2, 102, 10)
    expected = np.array([0.7978845608, 0.9775593519, 0.9881702533,
                         0.9919693005, 0.9939215919, 0.9951103466,
                         0.9959102089, 0.9964851811, 0.9969184165,
                         0.9972565726])
    result = _ewma._c4(n)

    np.testing.assert_allclose(expected, result)


def test__sd_sample(x):
    expected = 0.3415928
    result = _ewma._sd_sample(x)
    assert abs(expected - result) < 1e-5


def test__ewma_smooth(x):
    expected = np.array([33.56867, 33.46493, 33.57195, 33.61956, 33.58765,
                         33.67412, 33.67529, 33.59423, 33.57339, 33.49871,
                         33.52297, 33.41837, 33.44270, 33.37816, 33.47053])
    result = _ewma._ewma_smooth(x, x.mean(axis=0), lambda_=0.2)

    np.testing.assert_allclose(expected, result, 1e-5)


def test__ewma_boundary(x):
    ucl = np.array([33.75346, 33.81804, 33.85280, 33.87323,
                    33.88571, 33.89347, 33.89835, 33.90145,
                    33.90341, 33.90466, 33.90546, 33.90598,
                    33.90630, 33.90651, 33.90664])
    lcl = np.array([33.29320, 33.22862, 33.19387, 33.17344,
                    33.16096, 33.15320, 33.14831, 33.14522,
                    33.14326, 33.14200, 33.14120, 33.14069,
                    33.14036, 33.14016, 33.14002])
    center = x.mean(axis=0)
    stddev = _ewma._sd_moving_range(x, k=2)
    result = _ewma._ewma_boundary(x, stddev,
                                 crit=2.7, lambda_=0.2)

    np.testing.assert_allclose(ucl, center + result, 1e-5)
    np.testing.assert_allclose(lcl, center - result, 1e-5)


def test_EWMA(x):
    # NumPy
    result = _ewma.ewma(x, crit=2.7, lambda_=0.2, std_type='SD')
    assert (result.score - 33.67529) < 1e-5
    assert result.signif == False

    # Pandas and xarray
    for dt in (pd.Series, xr.DataArray):
        result = _ewma.ewma(dt(x), crit=2.7, lambda_=0.2, std_type='MR')
        assert result.signif == False

    # Test MAD
    result = _ewma.ewma(x, crit=2.7, lambda_=0.2, std_type='MAD')
    assert result.signif == False
