""" Result post-processing utilities

Includes comission and omission tests and robust linear model result
calculations
"""
import logging

import numpy as np
import numpy.lib.recfunctions as nprf
import scipy.stats
import statsmodels.api as sm

from ..regression.diagnostics import rmse
from ..utils import date2index

logger = logging.getLogger('yatsm')


# POST-PROCESSING
def commission_test(yatsm, alpha=0.10):
    """ Merge adjacent records based on Chow Tests for nested models

    Use Chow Test to find false positive, spurious, or unnecessary breaks
    in the timeseries by comparing the effectiveness of two separate
    adjacent models with one single model that spans the entire time
    period.

    Chow test is described:

    .. math::
        \\frac{[RSS_r - (RSS_1 + RSS_2)] / k}{(RSS_1 + RSS_2) / (n - 2k)}

    where:

        - :math:`RSS_r` is the RSS of the combined, or, restricted model
        - :math:`RSS_1` is the RSS of the first model
        - :math:`RSS_2` is the RSS of the second model
        - :math:`k` is the number of model parameters
        - :math:`n` is the number of total observations

    Because we look for change in multiple bands, the RSS used to compare
    the unrestricted versus restricted models is the mean RSS
    values from all ``model.test_indices``.

    Args:
        yatsm (YATSM model): fitted YATSM model to check for commission errors
        alpha (float): significance level for F-statistic (default: 0.10)

    Returns:
        np.ndarray: updated copy of ``yatsm.record`` with spurious models
            combined into unified model

    """
    if yatsm.record.size == 1:
        return yatsm.record

    k = yatsm.record[0]['coef'].shape[0]

    # Allocate memory outside of loop
    m_1_rss = np.zeros(yatsm.test_indices.size)
    m_2_rss = np.zeros(yatsm.test_indices.size)
    m_r_rss = np.zeros(yatsm.test_indices.size)

    models = []
    merged = False
    for i in range(len(yatsm.record) - 1):
        if merged:
            m_1 = models[-1]
        else:
            m_1 = yatsm.record[i]
        m_2 = yatsm.record[i + 1]

        # Unrestricted model starts/ends
        m_1_start = date2index(yatsm.dates, m_1['start'])
        m_1_end = date2index(yatsm.dates, m_1['end'])
        m_2_start = date2index(yatsm.dates, m_2['start'])
        m_2_end = date2index(yatsm.dates, m_2['end'])
        # Restricted start/end
        m_r_start = m_1_start
        m_r_end = m_2_end

        # Need enough obs to fit models (n > k)
        if (m_1_end - m_1_start) <= k or (m_2_end - m_2_start) <= k:
            logger.debug('Too few obs (n <= k) to merge segment')
            merged = False
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            continue

        n = m_r_end - m_r_start
        F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)

        for i_b, b in enumerate(yatsm.test_indices):
            m_1_rss[i_b] = np.linalg.lstsq(yatsm.X[m_1_start:m_1_end, :],
                                           yatsm.Y[b, m_1_start:m_1_end])[1]
            m_2_rss[i_b] = np.linalg.lstsq(yatsm.X[m_2_start:m_2_end, :],
                                           yatsm.Y[b, m_2_start:m_2_end])[1]
            m_r_rss[i_b] = np.linalg.lstsq(yatsm.X[m_r_start:m_r_end, :],
                                           yatsm.Y[b, m_r_start:m_r_end])[1]

        # Collapse RSS across all test indices for F statistic
        F = (
            ((m_r_rss.mean() - (m_1_rss.mean() + m_2_rss.mean())) / k) /
            ((m_1_rss.mean() + m_2_rss.mean()) / (n - 2 * k))
        )
        if F > F_crit:
            # Reject H0 and retain change
            # Only add in previous model if first model
            if i == 0:
                models.append(m_1)
            models.append(m_2)
            merged = False
        else:
            # Fail to reject H0 -- ignore change and merge
            m_new = np.copy(yatsm.record_template)[0]

            # Remove last previously added model from list to merge
            if i != 0:
                del models[-1]

            m_new['start'] = m_1['start']
            m_new['end'] = m_2['end']
            m_new['break'] = m_2['break']

            # Re-fit models and copy over attributes
            yatsm.fit_models(yatsm.X[m_r_start:m_r_end, :],
                             yatsm.Y[:, m_r_start:m_r_end])
            for i_m, _m in enumerate(yatsm.models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

            if 'magnitude' in yatsm.record.dtype.names:
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
            (model.dates >= min(r['start'], r['end'])) &
            (model.dates <= max(r['end'], r['start'])))[0]
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


def refit_record(model, prefix, estimator,
                 fitopt=None, keep_regularized=False):
    """ Refit YATSM model segments with a new estimator and update record

    YATSM class model must be ran and contain at least one record before this
    function is called.

    Args:
        model (YATSM model): YATSM model to refit
        prefix (str): prefix for refitted coefficient and RMSE (don't include
            underscore as it will be added)
        estimator (object): instance of a scikit-learn compatible estimator
            object
        fitopt (dict, optional): dict of options for the ``fit`` method of the
            ``estimator`` provided (default: None)
        keep_regularized (bool, optional): do not use features with coefficient
            estimates that are fit to 0 (i.e., if using L1 regularization)

    Returns:
        np.array: updated model.record NumPy structured array with refitted
            coefficients and RMSE

    """
    if not model:
        return None

    fitopt = fitopt or {}

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
                    refit[i_rec][refit_rmse][:] = rec['rmse']
                    continue
            else:
                nonzero = np.arange(n_series)

            # Fit
            estimator.fit(X[:, nonzero], y, **fitopt)
            # Store updated coefficients
            refit[i_rec][refit_coef][nonzero, i_y] = estimator.coef_
            refit[i_rec][refit_coef][0, i_y] += getattr(
                estimator, 'intercept_', 0.0)

            # Update RMSE
            refit[i_rec][refit_rmse][i_y] = rmse(
                y, estimator.predict(X[:, nonzero]))

    # Merge
    refit = nprf.merge_arrays((model.record, refit), flatten=True)

    return refit
