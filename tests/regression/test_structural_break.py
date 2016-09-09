""" Tests for yatsm.regression.structural_break
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

try:
    from rpy2 import robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    pandas2ri.activate()
except ImportError:
    has_rpy2 = False
else:
    has_rpy2 = True

from yatsm.regression.structural_break import cusum_OLS


@pytest.fixture('module')
def test_data():
    """ A very silly regression with an obvious change point
    """
    np.random.seed(0)
    y = np.concatenate((np.random.rand(250), np.random.rand(250) + 1))
    df = pd.DataFrame(np.column_stack((np.ones(500), np.arange(500), y)),
                      columns=['int', 'slope', 'y'])

    return df


@pytest.fixture('module')
def rpy2_strucchange():
    if has_rpy2:
        try:
            utils = importr('utils')
            utils.install_packages(
                'strucchange',
                repos='http://cran.revolutionanalytics.com/'
            )
        except Exception as exc:
            pytest.skip('Unable to install "strucchange"')
        else:
            return True
    return False


@pytest.fixture('module')
def strucchange_cusum_OLS(rpy2_strucchange, test_data):
    if has_rpy2 and rpy2_strucchange:
        try:
            rstr = """
            function(dat) {
                library(strucchange)
                model <- efp(y ~ ., data=dat, type="OLS-CUSUM")
                sctest(model)
            }
            """
            rfunc = robjects.r(rstr)
            rdat = robjects.r.ts(test_data)
            result = rfunc(rdat)
            result = dict(zip(result.names, list(result)))
        except Exception as exc:
            pass
        else:
            results = (result['statistic'][0], result['p.value'][0])
            return results

    # Validated against strucchange==1.5.1
    return (3.706774647551639, 2.325251102774928e-12)


def test_cusum_OLS(test_data, strucchange_cusum_OLS):
    """ Tested against strucchange 1.5.1
    """
    y = test_data.pop('y')
    X = test_data
    # Test sending pandas
    result = cusum_OLS(X, y)
    assert np.allclose(result.score, strucchange_cusum_OLS[0])
    assert np.allclose(result.pvalue, strucchange_cusum_OLS[1])

    # And ndarray and xarray
    result = cusum_OLS(X.values, xr.DataArray(y, dims=['time']))
    assert np.allclose(result.score, strucchange_cusum_OLS[0])
    assert np.allclose(result.pvalue, strucchange_cusum_OLS[1])
