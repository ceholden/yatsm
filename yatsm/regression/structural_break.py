# -*- coding: utf-8 -*-
""" Methods for estimating structural breaks in time series regressions

TODO: extract and move Chow test from "commission test" over to here
"""
from collections import namedtuple
import logging

import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr

from ..accel import try_jit

logger = logging.getLogger(__name__)

pandas_like = (pd.DataFrame, pd.Series, xr.DataArray)

# tuple: CUSUM-OLS results
CUSUMOLSResult = namedtuple('CUSUMOLSResult', ['index', 'score', 'cusum',
                                               'pvalue', 'signif'])

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
    score = _process.max()
    idx = _process.argmax()

    return process, score, idx


def cusum_OLS(X, y, alpha=0.05):
    u""" CUSUM-OLS test for structural breaks

    # TODO: same function for cusum_REC?

    Args:
        X (array like): 2D (n_features x n_obs) design matrix
        y (array like): 1D (n_obs) indepdent variable
        alpha (float): Test threshold (either 0.01, 0.05, or 0.10) from
            Ploberger and Krämer (1992)

    Returns:
        tuple: the change point (index of ``y``), the test pvalue, and
            a boolean testing if the CUSUM score is significant at the given
            ``alpha``
    """
    _X = X.values if isinstance(X, pandas_like) else X
    _y = y.values if isinstance(y, pandas_like) else y

    cusum, score, idx = _cusum_OLS(_X, _y)
    if isinstance(y, (pd.Series, pd.DataFrame)):
        idx = y.index[idx]
    elif isinstance(y, xr.DataArray):
        idx = y.to_series().index[idx]

    # crit = stats.kstwobign.isf(alpha)  ~70usec
    crit = CUSUM_OLS_CRIT[alpha]
    pval = stats.kstwobign.sf(score)

    return CUSUMOLSResult(index=idx, score=score, cusum=cusum,
                          pvalue=pval, signif=pval < crit)
