""" Result post-processing utilities

Includes comission and omission tests and robust linear model result
calculations
"""
import numpy as np
import numpy.lib.recfunctions
import scipy.stats

from ..regression import robust_fit as rlm
from ..utils import date2index


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

    where
        - :math:`RSS_u` is the RSS of the combined, or, restricted model
        - :math:`RSS_1` is the RSS of the first model
        - :math:`RSS_2` is the RSS of the second model
        - :math:`k` is the number of model parameters
        - :math:`n` is the number of total observations

    Because we look for change in multiple bands, the RSS used to compare
    the unrestricted versus restricted models is the L2 norm of RSS
    values from `self.test_indices`.

    Args:
      alpha (float): significance level for F-statistic (default: 0.01)

    Returns:
      np.ndarray: updated copy of `self.models` with spurious models
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
            m_1 = self.record[i]
        m_2 = model.record[i + 1]

        m_1_start = date2index(model.X[:, model._jx], m_1['start'])
        m_1_end = date2index(model.X[:, model._jx], m_1['end'])
        m_2_start = date2index(model.X[:, model._jx], m_2['start'])
        m_2_end = date2index(model.X[:, model._jx], m_2['end'])

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

            _models = model.fit_models(
                model.X[m_r_start:m_r_end, :],
                model.Y[:, m_r_start:m_r_end])

            for i_m, _m in enumerate(_models):
                m_new['coef'][:, i_m] = _m.coef
                m_new['rmse'][i_m] = _m.rmse

            # Preserve magnitude from 2nd model that was merged
            m_new['magnitude'] = m_2['magnitude']

            models.append(m_new)

            merged = True

    return np.array(models)


def omission_test(model, crit=0.05, behavior='ANY',
                  indices=None):
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

    omission = np.zeros((model.record.size, len(indices)),
                        dtype=bool)

    for i, r in enumerate(model.record):
        # Skip if no model fit
        if r['start'] == 0 or r['end'] == 0:
            continue
        # Find matching X and Y in data
        index = np.where(
            (model.X[:, model._jx] >= min(r['start'], r['end'])) &
            (model.X[:, model._jx] <= max(r['end'], r['start'])))[0]
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


def robust_record(model):
    """ Returns a copy of YATSM record output with robustly fitted models
    using non-zero coefficients from original regression.

    The returned model results should be more representative of the
    signal found because it will remove influence of outlying observations,
    such as clouds or shadows.

    If YATSM has not yet been run, returns None
    """
    if not model.ran:
        return None

    # Create new array for robust coefficients and RMSE
    robust = np.zeros(model.record.shape[0], dtype=[
        ('robust_coef', 'float32', (model.n_coef, len(model.fit_indices))),
        ('robust_rmse', 'float32', len(model.fit_indices)),
    ])

    # Update to robust model
    for i, r in enumerate(model.record):
        # Find matching X and Y in data
        index = np.where(
            (model.X[:, model._jx] >= min(r['start'], r['end'])) &
            (model.X[:, model._jx] <= max(r['end'], r['start'])))[0]
        # Grab matching X and Y
        _X = model.X[index, :]
        _Y = model.Y[:, index]

        # Refit each band
        for i_b, b in enumerate(model.fit_indices):
            # Find nonzero
            nonzero = np.nonzero(model.record[i]['coef'][:, i_b])[0]

            if nonzero.size == 0:
                continue

            # Setup model
            rirls_model = rlm.RLM(_Y[b, :], _X[:, nonzero], M=rlm.bisquare)

            # Fit
            fit = rirls_model.fit()
            # Store updated coefficients
            robust[i]['robust_coef'][nonzero, i_b] = fit.coefs

            # Update RMSE
            robust[i]['robust_rmse'][i_b] = \
                math.sqrt(rirls_model.rss / index.size)

        logger.debug('Updated record %s to robust results' % i)

    # Merge
    robust_record = np.lib.recfunctions.merge_arrays((model.record, robust),
                                                     flatten=True)

    return robust_record
