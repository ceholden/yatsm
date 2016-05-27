from __future__ import division

import numpy as np
import statsmodels.api as sm

from .accel import try_jit
from .regression import robust_fit as rlm

ndays = 365.25


@try_jit()  # np.array prevents nopython
def multitemp_mask(x, Y, n_year, crit=400,
                   green=1, swir1=4,
                   maxiter=10):
    """ Multi-temporal masking using RLM

    Taken directly from CCDC (Zhu and Woodcock, 2014). This "temporal masking"
    procedure was ported from CCDC v9.3.

    Args:
        x (ndarray): array of ordinal dates
        Y (ndarray): matrix of observed spectra
        n_year (float): "number of years to mask"
        crit (float): critical value for masking clouds/shadows
        green (int): 0 indexed value for green band in Y
            (default: 1)
        swir1 (int): 0 indexed value for SWIR (~1.55-1.75um) band
            in Y (default: 4)
        maxiter (int): maximum iterations for RLM fit

    Returns:
        mask (np.ndarray): mask where False indicates values to be masked

    """
    green = Y[green, :]
    swir1 = Y[swir1, :]

    n_year = np.ceil(n_year)
    w = 2.0 * np.pi / ndays

    X = np.array([np.ones_like(x),
                  np.cos(w * x),
                  np.sin(w * x),
                  np.cos(w / n_year * x),
                  np.sin(w / n_year * x)]).T

    green_RLM = rlm.RLM(M=rlm.bisquare, maxiter=maxiter).fit(X, green)
    swir1_RLM = rlm.RLM(M=rlm.bisquare, maxiter=maxiter).fit(X, swir1)

    mask = ((green - green_RLM.predict(X) < crit) *
            (swir1 - swir1_RLM.predict(X) > -crit))

    return mask


def smooth_mask(x, Y, span, crit=400, green=1, swir1=4,
                maxiter=5):
    """ Multi-temporal masking using LOWESS

    Taken directly from newer version of CCDC than Zhu and Woodcock, 2014.
    This "temporal masking" replaced the older method which used robust
    linear models. This version uses a regular LOWESS instead of robust
    LOWESS

    .. note::

        "span" argument is the inverse of "frac" from statsmodels and is
        actually 'k' in their code:

        `n = x.shape[0]`
        `k = int(frac * n + 1e-10)`

    .. todo::

        We need to put the data on a regular period since span changes as
        is right now. Statsmodels will only allow for dropna, so we would
        need to impute missing data somehow...

    Args:
        x (np.ndarray): array of ordinal dates
        Y (np.ndarray): matrix of observed spectra
        span (int): span of LOWESS
        crit (float): critical value for masking clouds/shadows
        green (int): 0 indexed value for green band in Y (default: 1)
        swir1 (int): 0 indexed value for SWIR (~1.55-1.75um) band
            in Y (default: 4)
        maxiter (int): maximum increases to span when checking for
            NaN in LOWESS results

    Returns:
      mask (ndarray): mask where False indicates values to be masked

    """
    # Reverse span to get frac
    frac = span / x.shape[0]
    # Estimate delta as "good choice": delta = 0.01 * range(exog)
    delta = (x.max() - x.min()) * 0.01

    # Run LOWESS checking for NaN in output
    i = 0
    green_lowess, swir1_lowess = np.nan, np.nan
    while (np.any(np.isnan(green_lowess)) or
           np.any(np.isnan(swir1_lowess))) and i < maxiter:
        green_lowess = sm.nonparametric.lowess(Y[green, :], x,
                                               frac=frac, delta=delta)
        swir1_lowess = sm.nonparametric.lowess(Y[swir1, :], x,
                                               frac=frac, delta=delta)
        span += 1
        frac = span / x.shape[0]
        i += 1

    mask = (((Y[green, :] - green_lowess[:, 1]) < crit) *
            ((Y[swir1, :] - swir1_lowess[:, 1]) > -crit))

    return mask
