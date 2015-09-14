""" Result post-processing utilities

Includes comission and omission tests and robust linear model result
calculations
"""
import logging
import math

import numpy as np
import numpy.lib.recfunctions as nprf
import scipy.stats
import sklearn
import statsmodels.api as sm

from ..regression import robust_fit as rlm
from ..utils import date2index

logger = logging.getLogger('yatsm')


# POST-PROCESSING
def commission_test(model, alpha=0.001):
    """ Merge adjacent records based on Chow Tests for nested models

    Use Chow Test to find false positive, spurious, or unnecessary breaks
    in the timeseries by comparing the effectiveness of two separate
    adjacent models with one single model that spans the entire time
    period.

    Chow test is described:

    .. math::
        \\frac{[RSS_r - (RSS_1 + RSS_2)] / k}{(RSS_1 + RSS_2) / (n - 2k)}

    where:

        - :math:`RSS_u` is the RSS of the combined, or, restricted model
        - :math:`RSS_1` is the RSS of the first model
        - :math:`RSS_2` is the RSS of the second model
        - :math:`k` is the number of model parameters
        - :math:`n` is the number of total observations

    Because we look for change in multiple bands, the RSS used to compare
    the unrestricted versus restricted models is the L2 norm of RSS
    values from `model.test_indices`.

    Args:
      alpha (float): significance level for F-statistic (default: 0.01)

    Returns:
      np.ndarray: updated copy of `model.record` with spurious models
        combined into unified model

    """
    if model.record.size == 1:
        return model.record

    k = model.n_coef

    models = []
    merged = False

    for i in range(len(model.record) - 1):
        if merged:
            m_1 = models[-1]
        else:
            m_1 = model.record[i]
        m_2 = model.record[i + 1]

        m_1_start = date2index(model.X[:, model.i_x], m_1['start'])
        m_1_end = date2index(model.X[:, model.i_x], m_1['end'])
        m_2_start = date2index(model.X[:, model.i_x], m_2['start'])
        m_2_end = date2index(model.X[:, model.i_x], m_2['end'])

        m_r_start = m_1_start
        m_r_end = m_2_end

        n = m_r_end - m_r_start

        F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)

        m_1_rss = np.zeros(model.test_indices.size)
        m_2_rss = np.zeros(model.test_indices.size)
        m_r_rss = np.zeros(model.test_indices.size)

        for i_b, b in enumerate(model.test_indices):
            m_1_rss[i_b] = scipy.linalg.lstsq(
                model.X[m_1_start:m_1_end, :],
                model.Y[b, m_1_start:m_1_end])[1]
            m_2_rss[i_b] = scipy.linalg.lstsq(
                model.X[m_2_start:m_2_end, :],
                model.Y[b, m_2_start:m_2_end])[1]
            m_r_rss[i_b] = scipy.linalg.lstsq(
                model.X[m_r_start:m_r_end, :],
                model.Y[b, m_r_start:m_r_end])[1]

        m_1_rss = np.linalg.norm(m_1_rss)
        m_2_rss = np.linalg.norm(m_2_rss)
        m_r_rss = np.linalg.norm(m_r_rss)

        F = ((m_r_rss - (m_1_rss + m_2_rss)) / k) / \
            ((m_1_rss + m_2_rss) / (n - 2 * k))

        if F > F_crit:
            # Reject H0 and retain change
            # Only add in previous model if first model
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            merged = False
        else:
            # Fail to reject H0 -- ignore change and merge
            m_new = np.copy(model.record_template)[0]

            # Remove last previously added model from list to merge
            if i != 0:
                del models[-1]

            m_new['start'] = m_1['start']
            m_new['end'] = m_2['end']
            m_new['break'] = m_2['break']

            _models = model.fit_models(model.X[m_r_start:m_r_end, :],
                                       model.Y[:, m_r_start:m_r_end])

            for i_m, _m in enumerate(_models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

            # Preserve magnitude from 2nd model that was merged
            m_new['magnitude'] = m_2['magnitude']

            models.append(m_new)

            merged = True

    return np.array(models)


def omission_test(model, crit=0.05, behavior='ANY', indices=None):
    """ Add omitted breakpoint into records based on residual stationarity

    Uses recursive residuals within a CUMSUM test to check if each model
    has omitted a "structural change" (e.g., land cover change). Returns
    an array of True or False for each timeseries segment record depending
    on result from `statsmodels.stats.diagnostic.breaks_cusumolsresid`.

    Args:
      crit (float, optional): Critical p-value for rejection of null
        hypothesis that data contain no structural change
      behavior (str, optional): Method for dealing with multiple
        `test_indices`. `ANY` will return True if any one test index
        rejects the null hypothesis. `ALL` will only return True if ALL
        test indices reject the null hypothesis.
      indices (np.ndarray, optional): Array indices to test. User provided
        indices must be a subset of `model.test_indices`.

    Returns:
      np.ndarray: Array of True or False for each record where
        True indicates omitted break point

    """
    if behavior.lower() not in ['any', 'all']:
        raise ValueError('`behavior` must be "any" or "all"')

    if not indices:
        indices = model.test_indices

    if not np.all(np.in1d(indices, model.test_indices)):
        raise ValueError('`indices` must be a subset of '
                         '`model.test_indices`')

    if not model.ran:
        return np.empty(0, dtype=bool)

    omission = np.zeros((model.record.size, len(indices)), dtype=bool)

    for i, r in enumerate(model.record):
        # Skip if no model fit
        if r['start'] == 0 or r['end'] == 0:
            continue
        # Find matching X and Y in data
        index = np.where(
            (model.X[:, model.i_x] >= min(r['start'], r['end'])) &
            (model.X[:, model.i_x] <= max(r['end'], r['start'])))[0]
        # Grab matching X and Y
        _X = model.X[index, :]
        _Y = model.Y[:, index]

        for i_b, b in enumerate(indices):
            # Create OLS regression
            ols = sm.OLS(_Y[b, :], _X).fit()
            # Perform CUMSUM test on residuals
            test = sm.stats.diagnostic.breaks_cusumolsresid(
                ols.resid, _X.shape[1])

            if test[1] < crit:
                omission[i, i_b] = True
            else:
                omission[i, i_b] = False

    # Collapse band answers according to `behavior`
    if behavior.lower() == 'any':
        return np.any(omission, 1)
    else:
        return np.all(omission, 1)


def refit_record(model, prefix, predictor, keep_regularized=False):
    """ Refit YATSM model segments with a new predictor and update record

    YATSM class model must be ran and contain at least one record before this
    function is called.

    Args:
        model (YATSM model): YATSM model to refit
        prefix (str): prefix for refitted coefficient and RMSE (don't include
            underscore as it will be added)
        predictor (object): instance of a scikit-learn compatible prediction
            object
        keep_regularized (bool, optional): do not use features with coefficient
            estimates that are fit to 0 (i.e., if using L1 regularization)

    Returns:
        np.array: updated model.record NumPy structured array with refitted
            coefficients and RMSE

    """
    if not model:
        return None

    refit_coef = prefix + '_coef'
    refit_rmse = prefix + '_rmse'

    # Create new array for robust coefficients and RMSE
    n_coef, n_series = model.record[0]['coef'].shape
    refit = np.zeros(model.record.shape[0], dtype=[
        (refit_coef, 'float32', (n_coef, n_series)),
        (refit_rmse, 'float32', (n_series)),
    ])

    for i_rec, rec in enumerate(model.record):
        # Find matching X and Y in data
        # start/end dates are considered in case ran backward
        index = np.where((model.dates >= min(rec['start'], rec['end'])) &
                         (model.dates <= max(rec['start'], rec['end'])))[0]

        X = model.X.take(index, axis=0)
        Y = model.Y.take(index, axis=1)

        # Refit each band
        for i_y, y in enumerate(Y):
            if keep_regularized:
                # Find nonzero in case of regularized regression
                nonzero = np.nonzero(rec['coef'][:, i_y])[0]
                if nonzero.size == 0:
                    refit[i_rec]['rmse'] = rec['rmse']
                    continue
            else:
                nonzero = np.arange(n_series)

            # Fit
            lm = sklearn.clone(predictor)
            lm = lm.fit(X[:, nonzero], y)
            # Store updated coefficients
            refit[i_rec][refit_coef][nonzero, i_y] = lm.coef_

            # Update RMSE
            refit[i_rec][refit_rmse][i_y] = \
                ((y - lm.predict(X[:, nonzero])) ** 2).mean() ** 0.5

    # Merge
    refit = nprf.merge_arrays((model.record, refit), flatten=True)

    return refit
