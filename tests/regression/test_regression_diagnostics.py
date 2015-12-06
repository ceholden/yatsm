import numpy as np
import pytest

from yatsm.regression.diagnostics import rmse

n = 500


@pytest.fixture
def y(prng):
    y = prng.standard_normal(n)
    return y


def test_rmse(y):
    yhat = y + 1.0
    _rmse = rmse(y, yhat)
    np.testing.assert_allclose(_rmse, 1.0)
