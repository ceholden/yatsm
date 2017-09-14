""" Tests for yatsm.regression.robust_fit
"""
import numpy as np
import pytest

import yatsm.regression.robust_fit as rf


@pytest.mark.parametrize(('array', 'ans', ), [
    (np.array([1, 1, 2, 2, 4, 6, 9, ]), 1.4825796)  # from Wikipedia example
])
def test_mad(array, ans):
    mad = rf.mad(array)
    assert (mad - ans) < 1e-7


def test_RLM(X_y_intercept_slope):
    X, y, intercept, slope = X_y_intercept_slope
    m = rf.RLM(M=rf.bisquare, tune=4.685,
               scale_est=rf.mad, scale_constant=0.6745, update_scale=True,
               maxiter=50, tol=1e-8)
    m.fit(X, y)

    np.testing.assert_allclose(m.coef_, [intercept, slope])
    np.testing.assert_allclose(m.predict(X), X[:, 1] * slope + intercept)


@pytest.mark.parametrize(('X', 'y'), [
    (np.random.rand(n, n), np.random.rand(n))
    for n in range(1, 10)
])
def test_RLM_issue88(X, y):
    """ Issue 88: Numeric problems when n_obs == n_reg/k/p/number of regressors

    The regression result will be garbage so we're not worrying about the
    coefficients. However, it shouldn't raise an exception.
    """
    m = rf.RLM(M=rf.bisquare, tune=4.685,
               scale_est=rf.mad, scale_constant=0.6745, update_scale=True,
               maxiter=50, tol=1e-8)
    m.fit(X, y)
    m.predict(X)


@pytest.fixture
def X_y_intercept_slope(request):
    np.random.seed(0)
    slope, intercept = 2., 5.
    X = np.c_[np.ones(10), np.arange(10)]
    y = slope * X[:, 1] + intercept

    # Add noise
    y[9] = 0
    return X, y, intercept, slope
