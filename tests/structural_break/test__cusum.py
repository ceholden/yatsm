""" Tests for yatsm.regression.structural_break
"""
import numpy as np
import pandas as pd
import patsy
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

from yatsm.structural_break import _cusum as cu


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
            base = importr('base')
            utils = importr('utils')
            has_pkg = base.require('strucchange')[0]
            if not has_pkg:
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
    result = cu.cusum_OLS(X, y)
    assert np.allclose(result.score, strucchange_cusum_OLS[0])
    assert np.allclose(result.pvalue, strucchange_cusum_OLS[1])

    # And ndarray and xarray
    result = cu.cusum_OLS(X.values, xr.DataArray(y, dims=['time']))
    assert np.allclose(result.score, strucchange_cusum_OLS[0])
    assert np.allclose(result.pvalue, strucchange_cusum_OLS[1])


def test_cusum_rec_efp_sctest_pvalue(airquality):
    y = airquality['Ozone'].values
    X = patsy.dmatrix('1 + SolarR + Wind + Temp', data=airquality)

    process = cu._cusum_rec_efp(X, y)
    stat = cu._cusum_rec_sctest(process)
    pvalue = cu._brownian_motion_pvalue(0.2335143, 1)

    np.testing.assert_allclose(process.sum(), 1.9446869, rtol=1e-5)
    np.testing.assert_allclose(stat, 0.2335143, rtol=1e-5)
    np.testing.assert_allclose(pvalue, 0.9657902, rtol=1e-4)


@pytest.mark.parametrize(('alpha', 'truth'), [
    # pvalues are obtained from statsmodels and are estimates
    (0.10, 0.850),
    (0.05, 0.948),
    (0.01, 1.143)
])
def test_cusum_rec_test_crit(alpha, truth):
    test = cu._cusum_rec_test_crit(alpha)
    np.testing.assert_allclose(truth, test, rtol=1e-3)
