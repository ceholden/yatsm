# -*- coding: utf-8 -*-
""" Methods for estimating structural breaks in time series regressions

TODO: extract and move Chow test from "commission test" over to here
"""
from __future__ import division

from collections import namedtuple
import logging

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy import stats
from scipy.stats import norm
import xarray as xr

from ._core import pandas_like, StructuralBreakResult
from ..accel import try_jit
from ..regression._recresid import _recresid

logger = logging.getLogger(__name__)

pnorm = norm.cdf


# OLS-CUSUM
# dict: CUSUM OLS critical values
CUSUM_OLS_CRIT = {
    0.01: 1.63,
    0.05: 1.36,
    0.10: 1.22
}

@try_jit(nopython=True, nogil=True)
def _cusum(resid, ddof):
    n = resid.size
    df = n - ddof

    sigma = ((resid ** 2).sum() / df * n) ** 0.5
    process = resid.cumsum() / sigma
    return process


@try_jit(nopython=True, nogil=True)
def _cusum_OLS(X, y):
    n, p = X.shape
    beta = np.linalg.lstsq(X, y)[0]
    resid = np.dot(X, beta) - y

    process = _cusum(resid, p)
    _process = np.abs(process)
    idx = _process.argmax()
    score = _process[idx]

    return process, score, idx


def cusum_OLS(X, y, alpha=0.05):
    ur""" OLS-CUSUM test for structural breaks

    Tested against R's ``strucchange`` package and is faster than
    the equivalent function in the ``statsmodels`` Python package when
    Numba is installed.

    The OLS-CUSUM test statistic, based on a single OLS regression, is defined
    as:

    .. math::

        W_n^0(t) = \frac{1}{\hat{\sigma}\sqrt{n}}
                   \sum_{i=1}^{n}{\hat{\mu_i}}

    Args:
        X (array like): 2D (n_obs x n_features) design matrix
        y (array like): 1D (n_obs) indepdent variable
        alpha (float): Test threshold (either 0.01, 0.05, or 0.10) from
            Ploberger and Krämer (1992)

    Returns:
        StructuralBreakResult: A named tuple include the the test name,
        change point (index of ``y``), the test ``score`` and ``pvalue``,
        and a boolean testing if the CUSUM score is
        significant at the given ``alpha``

    """
    _X = X.values if isinstance(X, pandas_like) else X
    _y = y.values.ravel() if isinstance(y, pandas_like) else y.ravel()

    process, score, idx = _cusum_OLS(_X, _y)
    if isinstance(y, pandas_like):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            index = y.index
            idx = index[idx]
        elif isinstance(y, xr.DataArray):
            index = y.to_series().index
            idx = index[idx]
        process = pd.Series(data=process, index=index, name='OLS-CUSUM')

    # crit = stats.kstwobign.isf(alpha)  ~70usec
    crit = CUSUM_OLS_CRIT[alpha]
    pval = stats.kstwobign.sf(score)

    return StructuralBreakResult(method='OLS-CUSUM',
                                 index=idx,
                                 score=score,
                                 process=process,
                                 boundary=crit,
                                 pvalue=pval,
                                 signif=score > crit)


# REC-CUSUM
def _brownian_motion_pvalue(x, k):
    """ Return pvalue for some given test statistic """
    # TODO: Make generic, add "type='Brownian Motion'"?
    if x < 0.3:
        p = 1 - 0.1464 * x
    else:
        p = 2 * (1 -
                 pnorm(3 * x) +
                 np.exp(-4 * x ** 2) * (pnorm(x) + pnorm(5 * x) - 1) -
                 np.exp(-16 * x ** 2) * (1 - pnorm(x)))
    return 1 - (1 - p) ** k


def _cusum_rec_test_crit(alpha):
    """ Return critical test statistic value for some alpha """
    return brentq(lambda _x: _brownian_motion_pvalue(_x, 1) -  alpha, 0, 20)


@try_jit(nopython=True, nogil=True)
def _cusum_rec_boundary(x, alpha=0.05):
    """ Equivalent to ``strucchange::boundary.efp``` for Rec-CUSUM """
    n = x.ravel().size
    bound = _cusum_rec_test_crit(alpha)
    boundary = (bound + (2 * bound * np.arange(0, n) / (n - 1)))

    return boundary


@try_jit()
def _cusum_rec_efp(X, y):
    """ Equivalent to ``strucchange::efp`` for Rec-CUSUM """
    # Run "efp"
    n, k = X.shape
    w = _recresid(X, y, k)[k:]
    sigma = w.var(ddof=1) ** 0.5  # can't jit because of ddof
    w = np.concatenate((np.array([0]), w))
    return np.cumsum(w) / (sigma * (n - k) ** 0.5)


@try_jit(nopython=True, nogil=True)
def _cusum_rec_sctest(x):
    """ Equivalent to ``strucchange::sctest`` for Rec-CUSUM """
    x = x[1:]
    j = np.linspace(0, 1, x.size + 1)[1:]
    x = x * 1 / (1 + 2 * j)
    stat = np.abs(x).max()

    return stat


def cusum_recursive(X, y, alpha=0.05):
    ur""" Rec-CUSUM test for structural breaks

    Tested against R's ``strucchange`` package.

    The REC-CUSUM test, based on the recursive residuals, is defined as:

    .. math::

        W_n(t) = \frac{1}{\tilde{\sigma}\sqrt{n}}
                 \sum_{i=k+1}^{k+(n-k)}{\tilde{\mu_i}}


    Critical values for this test statistic are taken from::

        A. Zeileis. p values and alternative boundaries for CUSUM tests.
            Working Paper 78, SFB "Adaptive Information Systems and Modelling
            in Economics and Management Science", December 2000b.

    Args:
        X (array like): 2D (n_obs x n_features) design matrix
        y (array like): 1D (n_obs) indepdent variable
        alpha (float): Test threshold

    Returns:
        StructuralBreakResult: A named tuple include the the test name,
        change point (index of ``y``), the test ``score`` and ``pvalue``,
        and a boolean testing if the CUSUM score is
        significant at the given ``alpha``

    """
    _X = X.values if isinstance(X, pandas_like) else X
    _y = y.values.ravel() if isinstance(y, pandas_like) else y.ravel()

    process = _cusum_rec_efp(_X, _y)
    stat = _cusum_rec_sctest(process)
    stat_pvalue = _brownian_motion_pvalue(stat, 1)

    pvalue_crit = _cusum_rec_test_crit(alpha)
    if stat_pvalue < alpha:
        boundary = _cusum_rec_boundary(process, alpha)
        idx = np.where(np.abs(process) > boundary)[0].min()
    else:
        idx = np.abs(process).max()

    if isinstance(y, pandas_like):
        if isinstance(y, (pd.Series, pd.DataFrame)):
            index = y.index
            idx = index[idx]
        elif isinstance(y, xr.DataArray):
            index = y.to_series().index
            idx = index[idx]
        process = pd.Series(data=process, index=index, name='REC-CUSUM')
        boundary = pd.Series(data=boundary, index=index, name='Boundary')

    return StructuralBreakResult(method='REC-CUSUM',
                                 process=process,
                                 boundary=boundary,
                                 index=idx,
                                 pvalue=stat_pvalue,
                                 score=stat,
                                 signif=stat_pvalue < pvalue_crit)
