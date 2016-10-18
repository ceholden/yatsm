import numpy as np
import pytest

from yatsm.regression.diagnostics import std, rmse

n = 500


@pytest.fixture
def y(prng):
    y = prng.standard_normal(n)
    return y


@pytest.mark.parametrize('n', [100, 1000, 10000])
def test_std(n):
    np.random.seed(42)
    x = np.random.normal(size=n)
    for ddof in (1, 2, 3, 4):
        result = std(x, ddof=ddof)
        expected = np.std(x, ddof=ddof)
        assert abs(result - expected) < 1e-4


def test_rmse(y):
    yhat = y + 1.0
    _rmse = rmse(y, yhat)
    np.testing.assert_allclose(_rmse, 1.0)
