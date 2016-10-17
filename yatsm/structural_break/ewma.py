""" Exponentially Weighted Moving Average (EWMA)
"""
from __future__ import division

import numpy as np
from scipy.special import gamma

from ..accel import try_jit


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


def _c4(n):
    """ Bias correction factor for normal distribution

    See: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Results_for_the_normal_distribution
    """
    return ((2 / (n - 1)) ** 0.5 *
            np.exp(_lgamma(n / 2) - _lgamma((n - 1) / 2)))


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


@try_jit(nopython=True, nogil=True)
def _ewma_boundary(n, mean, stddev, crit=3.0, lambda_=0.2):
    """ Calculate control chart boundaries for a given process

    Args:
        n (int): Number of observations in process sequence
        mean (float): Center point, or mean, of signal
        stddev (float): Standard deviation of the signal
        crit (float): Critical threshold for boundary, given as a scalar
            multiplier of the standard deviation
        lambda_ (float): "Memory" parameter, bound [0, 1]

    Returns:
        tuple(np.ndarray, np.ndarray): Upper and lower boundaries of process
    """
    x = np.arange(1, n + 1)
    cl = crit * stddev * np.sqrt(
        (lambda_ / (2 - lambda_)) * (1 - (1 - lambda_) ** (2 * x)))
    
    return (mean + cl, mean - cl)


@try_jit(nopython=True, nogil=True)
def _ewma_smooth(y, lambda_=0.2, start=None):
    if start is None:
        start = y[0]
    n = y.shape[0]

    S1 = np.eye(n)
    for i in range(n - 1):
        for j in range(i, n):
            S1[j, i] = (1 - lambda_) ** (j - i)
    S2 = (1 - lambda_) ** np.arange(1, n + 1)
    z = lambda_ * np.dot(S1, y) + S2 * start

    return z


@try_jit
def _ewma(y, lambda_=0.2):
    center = np.mean(y, axis=0) 


def ewma(y, lambda_=0.2):
    pass
