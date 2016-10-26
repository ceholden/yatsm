""" Exponentially Weighted Moving Average (EWMA)
"""
from __future__ import division

import numpy as np
import pandas as pd
from scipy.special import gamma
import xarray as xr

from ._core import StructuralBreakResult, pandas_like
from ..accel import try_jit
from ..regression.robust_fit import mad


def _rolling_window(a, window):
    """ Return a rolling window using NumPy strides

    Credit: http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html
    """
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def _lgamma(n):
    """ Natural log of absolute value of gamma function (``lgamma`` in R)
    """
    return np.log(np.abs(gamma(n)))


#: np.ndarray: Expected value of the sample range of ``n`` normally
#              distributed variables. Index on ``n - 1`` to retrieve the
#              value of ``d2(n)``
_d2_n = np.array([np.nan, np.nan, 1.128, 1.693, 2.059, 2.326, 2.534, 2.704,
                  2.847, 2.970, 3.078, 3.173, 3.258, 3.336, 3.407, 3.472,
                  3.532, 3.588, 3.640, 3.689, 3.735, 3.778, 3.819, 3.858,
                  3.895, 3.931])


@try_jit
def _sd_moving_range(y, k=2):
    """ Moving range estimate of standard deviation

    Args:
        y (np.ndarray): Data
        k (int): Number of observations included in moving range

    Returns:
        float: Estimated standard deviation
    """
    n = y.shape[0]
    d = np.ptp(_rolling_window(y, k), axis=1).sum()
    sd = (d / (n - k + 1)) / _d2_n[k]

    return sd


def _c4(n):
    """ Bias correction factor for normal distribution

    See: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Results_for_the_normal_distribution
    """
    return ((2 / (n - 1)) ** 0.5 *
            np.exp(_lgamma(n / 2) - _lgamma((n - 1) / 2)))


def _sd_sample(x):
    """ Sample standard deviation
    """
    n = x.shape[0]
    return np.std(x, ddof=1) / _c4(n)


@try_jit(nopython=True, nogil=True)
def _ewma_boundary(x, stddev, crit=3.0, lambda_=0.2):
    """ Calculate control chart boundaries for a given process

    Args:
        x (np.ndarray): Observations
        stddev (float): Standard deviation of the observations
        crit (float): Critical threshold for boundary, given as a scalar
            multiplier of the standard deviation
        lambda_ (float): "Memory" parameter, bound [0, 1]

    Returns:
        tuple(np.ndarray, np.ndarray): Upper and lower boundaries of process
    """
    n = x.shape[0]
    x = np.arange(1, n + 1)
    cl = crit * stddev * np.sqrt(
        (lambda_ / (2 - lambda_)) * (1 - (1 - lambda_) ** (2 * x)))

    return cl


@try_jit(nopython=True, nogil=True)
def _ewma_smooth(y, start, lambda_=0.2):
    n = y.shape[0]

    S1 = np.eye(n)
    for i in range(n - 1):
        for j in range(i, n):
            S1[j, i] = (1 - lambda_) ** (j - i)
    S2 = (1 - lambda_) ** np.arange(1, n + 1)
    z = lambda_ * np.dot(S1, y) + S2 * start

    return z


@try_jit
def _ewma(y, lambda_=0.2, crit=3.0, center=True, std_type='SD'):
    if center:
        _center = np.mean(y, axis=0)
    else:
        _center = 0.0
    if std_type == 'SD':
        sd = _sd_sample(y)
    elif std_type == 'MAD':
        sd = mad(y)
    else:
        sd = _sd_moving_range(y, k=2)
    process = _ewma_smooth(y, lambda_=lambda_, start=_center)
    boundary = _ewma_boundary(y, sd, crit=crit, lambda_=lambda_)
    violation = np.abs(process - _center) > boundary
    idx = np.where(violation)[0]
    if len(idx) != 0:
        idx = idx.min()
        score = process[idx]
        signif = True
    else:
        idx = np.abs(process).argmax()
        score = process[idx]
        signif = False

    return (process, boundary, score, idx, signif)


def ewma(y, lambda_=0.2, crit=3.0, center=True, std_type='SD'):
    """ Exponentially Weighted Moving Average test

    Args:
        y (array like): Time series to test. Should be sorted chronologically
        lambda_ (float): "Memory" parameter, bound [0, 1]
        crit (float): Critical threshold for boundary, given as a scalar
            multiplier of the standard deviation
        center (bool): Center time series before calculating EWMA
        std_type (str): Method for calculating process standard deviation.
            Calculated using:

            * ``MR`` for an estimate based on the "moving range" of
              the scaled mean
            * ``SD`` for the sample standard deviation
            * ``MAD`` for the Median Absolute Deviation estimate of
              standard deviation

    Returns:
        StructuralBreakResult: A named tuple include the the test name,
        change point (index of ``y``), the test ``score``,
        and a boolean testing if the EWMA score is significant at the
        given ``crit``

    """
    _y = y.values.ravel() if isinstance(y, pandas_like) else y.ravel()

    process, boundary, score, idx, signif = _ewma(_y,
                                                  lambda_=lambda_,
                                                  crit=crit,
                                                  center=center,
                                                  std_type=std_type)
    if isinstance(y, pandas_like):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            index = y.index
            idx = index[idx]
        elif isinstance(y, xr.DataArray):
            index = y.to_series().index
            idx = index[idx]
        process = pd.Series(data=process, index=index, name='EWMA')
        boundary = pd.Series(data=boundary, index=index, name='Boundary')

    return StructuralBreakResult(method='EWMA',
                                 process=process,
                                 boundary=boundary,
                                 index=idx,
                                 pvalue=None,
                                 score=score,
                                 signif=signif)
