from datetime import datetime as dt
import os

import numpy as np
import sklearn
import pytest

from yatsm.algorithms.yatsm import YATSM

here = os.path.dirname(__file__)


# REAL DATA
@pytest.fixture(scope='session')
def unmasked_ts(request):
    """ Return dict of an unmasked example timeseries

    Dict contains:
        dates: ordinal dates
        Y: observations of 7 bands + Fmask
        X: design matrix
        design_str: Patsy design specification
        design_dict: Patsy column name indices for design matrix X

    Mask based on Fmask values (retain 0 and 1) and optical data min/max
    of 0 to 10,000.
    """
    f = os.path.join(here, 'data', 'example_timeseries.npz')
    return np.load(f)


@pytest.fixture(scope='session')
def masked_ts(request):
    """ Return dict of a masked example timeseries

    Dict contains:
        dates: ordinal dates
        Y: observations of 7 bands + Fmask
        X: design matrix
        design_str: Patsy design specification
        design_dict: Patsy column name indices for design matrix X

    Mask based on Fmask values (retain 0 and 1) and optical data min/max
    of 0 to 10,000.
    """
    f = os.path.join(here, 'data', 'example_timeseries_masked.npz')
    return np.load(f)


# SIMULATED DATA
def setup_dummy_YATSM(X, Y, dates, i_breaks):
    """ Setup a dummy YATSM model

    Args:
        X (np.ndarray): n x p features
        Y (np.ndarray): n_series x n independent data
        dates (np.ndarray): n dates
        i_breaks (iterable): indices of ``dates`` representing break dates
            (can be zero or nonzero, but len(i_breaks) is len(yatsm.record))

    Returns:
        YATSM model
    """
    n = dates.size
    yatsm = YATSM()
    yatsm.X, yatsm.Y, yatsm.dates = X, Y, dates
    yatsm.n_coef, yatsm.n_series = X.shape[1], Y.shape[0]
    yatsm.models = np.array([sklearn.clone(yatsm.estimator)
                             for i in range(yatsm.n_series)])
    yatsm.test_indices = np.arange(yatsm.n_series)
    n_models = len(i_breaks)
    yatsm.record = np.hstack([yatsm.record_template] * n_models)

    def populate_record(yatsm, i_rec, i_start, i_end, i_break):
        yatsm.record[i_rec]['start'] = yatsm.dates[i_start]
        yatsm.record[i_rec]['end'] = yatsm.dates[i_end]
        yatsm.record[i_rec]['break'] = (yatsm.dates[i_break] if i_break
                                        else i_break)
        yatsm.fit_models(X[i_start:i_end, :], Y[:, i_start:i_end])
        for i, m in enumerate(yatsm.models):
            yatsm.record[i_rec]['coef'][:, i] = m.coef
            yatsm.record[i_rec]['rmse'][i] = m.rmse
        return yatsm

    i_start = 0
    i_end = i_breaks[0] - 1 if i_breaks[0] else n - 1
    i_break = i_breaks[0]
    yatsm = populate_record(yatsm, 0, i_start, i_end, i_break)

    for idx, i_break in enumerate(i_breaks[1:]):
        i_start = i_breaks[idx] + 1
        i_end = i_break - 1 if i_break else n - 1
        yatsm = populate_record(yatsm, idx + 1, i_start, i_end, i_break)

    return yatsm


def _sim_no_change_data():
    """ Return a simulated timeseries with no change
    """
    np.random.seed(123456789)
    dates = np.arange(dt.strptime('2000-01-01', '%Y-%m-%d').toordinal(),
                      dt.strptime('2005-01-01', '%Y-%m-%d').toordinal(),
                      16)
    n = dates.size
    X = np.column_stack((np.ones(n), dates))  # n x p
    _y = np.linspace(0, 10, n) + np.random.standard_normal(n)
    Y = np.array([_y] * 2)  # nseries x n
    return X, Y, dates


@pytest.fixture(scope='module')
def sim_nochange(request):
    """ Return a dummy YATSM model container with a no-change dataset

    "No-change" dataset is simply a timeseries drawn from samples of one
    standard normal.
    """
    X, Y, dates = _sim_no_change_data()
    return setup_dummy_YATSM(X, Y, dates, [0])


@pytest.fixture(scope='module')
def sim_no_real_change_1(request):
    """ Return a dummy YATSM model container with a spurious change

    "Spurious" dataset is simply a timeseries drawn from samples of one
    standard normal, but with a record indicating that there was a change.
    """
    X, Y, dates = _sim_no_change_data()
    n = dates.size
    # Put a break somewhere in the middle
    return setup_dummy_YATSM(X, Y, dates, [n // 2, 0])


@pytest.fixture(scope='module')
def sim_no_real_change_2(request):
    """ Return a dummy YATSM model container with two spurious changes

    "Spurious" dataset is simply a timeseries drawn from samples of one
    standard normal, but with a record indicating that there was a change.
    """
    X, Y, dates = _sim_no_change_data()
    n = dates.size
    # Put two breaks somewhere in the middle
    return setup_dummy_YATSM(X, Y, dates, [n // 4, n // 2, 0])


@pytest.fixture(scope='module')
def sim_real_change(request):
    """ Return a dummy YATSM model container with a real change

    "Real change" dataset is simply a timeseries drawn from samples of two
    normal distributions with greatly different mean values.
    """
    np.random.seed(123456789)
    dates = np.arange(dt.strptime('2000-01-01', '%Y-%m-%d').toordinal(),
                      dt.strptime('2005-01-01', '%Y-%m-%d').toordinal(),
                      16)
    n = dates.size
    X = np.column_stack((np.ones(n), dates))  # n x p
    n_1, n_2 = n // 2, n - n // 2
    _y1 = np.linspace(0, 10, n_1) + np.random.standard_normal(n_1)
    _y2 = np.linspace(20, 10, n_2) + np.random.standard_normal(n_2)
    Y = np.array([
        np.concatenate((_y1, _y2)),
        np.concatenate((_y1, _y2))
    ])  # nseries x n

    # Put a break somewhere in the middle
    return setup_dummy_YATSM(X, Y, dates, [n_1, 0])
