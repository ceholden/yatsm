""" Test CCDCesque results with some random parameters

Just a consistency/regression check for now to ensure nothing changes during
optimization, API changes, or refactoring attempts. The "truth" isn't
necessarily an acceptable answer from the algorithm.
"""
from datetime import datetime as dt

import numpy as np
import pytest
import sklearn.linear_model

from yatsm.algorithms.ccdc import CCDCesque


@pytest.fixture(scope='function')
def model():
    model = CCDCesque(
        test_indices=np.array([2, 3, 4, 5]),
        estimator=sklearn.linear_model.Lasso(alpha=20),
        consecutive=6,
        threshold=3.5,
        min_obs=24,
        min_rmse=100,
        retrain_time=365.25,
        screening='RLM',
        screening_crit=400.0,
        green_band=1,
        swir1_band=4,
        remove_noise=True,
        dynamic_rmse=True,
        slope_test=True,
        idx_slope=1
    )
    return model


@pytest.fixture(scope='function')
def record(masked_ts, model):
    X, Y, ordinal = masked_ts['X'], masked_ts['Y'], masked_ts['dates']
    record = model.fit(X, Y[:-1, :], ordinal)
    return record


def test_CCDCesque_changedates(record):
    # Test start, end, and break dates of first two segments
    starts = [dt.strptime('1984-06-04', '%Y-%m-%d'),
              dt.strptime('1999-07-16', '%Y-%m-%d')]
    ends = [dt.strptime('1999-06-30', '%Y-%m-%d'),
            dt.strptime('2003-07-11', '%Y-%m-%d')]
    breaks = [dt.strptime('1999-07-16', '%Y-%m-%d'),
              dt.strptime('2003-07-27', '%Y-%m-%d')]

    for i in range(2):
        assert dt.fromordinal(record[i]['start']) == starts[i]
        assert dt.fromordinal(record[i]['end']) == ends[i]
        assert dt.fromordinal(record[i]['break']) == breaks[i]


# And now for the more hazardous tests on coefficents and RMSE
def test_CCDCesque_coef(record):
    swir1_coefs = np.array([-81.63251495, -67.66514587, -74.57978058, -0.,
                            47.00354004, 10.3005724, 482.47711182])
    np.testing.assert_allclose(record[0]['coef'][4, :], swir1_coefs)


def test_CCDCesque_rmse(record):
    swir1_rmse = 77.21417999
    np.testing.assert_allclose(record[0]['rmse'][4], swir1_rmse)
