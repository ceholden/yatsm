""" Test postprocessing algorithms
"""
import numpy as np

from yatsm.algorithms.postprocess import commission_test, refit_record


# COMMISSION TEST
def test_commission_nochange(sim_nochange):
    """ In no change situation, we should get back exactly what we gave in
    """
    record = commission_test(sim_nochange, 0.10)
    assert len(record) == 1
    np.testing.assert_array_equal(record, sim_nochange.record)


def test_commission_no_real_change_1(sim_no_real_change_1):
    """ Test commission test's ability to resolve spurious change
    """
    record = commission_test(sim_no_real_change_1, 0.01)
    assert len(record) == 1
    assert record[0]['break'] == 0


def test_commission_no_real_change_2(sim_no_real_change_2):
    """ Test commission test's ability to resolve two spurious changes
    """
    record = commission_test(sim_no_real_change_2, 0.01)
    assert len(record) == 1
    assert record[0]['break'] == 0


def test_commission_real_change(sim_real_change):
    """ Test commission test's ability to avoid merging real changes

    This test is run with a relatively large p value (very likely to reject H0
    and retain changes)
    """
    record = commission_test(sim_real_change, 0.10)
    assert len(record) == len(sim_real_change.record)


# REFIT
def test_refit_nochange_rlm(sim_nochange):
    """ Test record refitting of one record using robust linear models
    """
    from yatsm.regression import RLM
    estimator = RLM(maxiter=10)

    refit = refit_record(sim_nochange, 'rlm', estimator,
                         keep_regularized=True)
    assert 'rlm_coef' in refit.dtype.names
    assert 'rlm_rmse' in refit.dtype.names

    coef = np.array([[-3.84164779e+03, -3.84164779e+03],
                     [5.26200993e-03, 5.26200993e-03]])
    rmse = np.array([0.96866816, 0.96866816])
    np.testing.assert_allclose(refit[0]['rlm_coef'], coef)
    np.testing.assert_allclose(refit[0]['rlm_rmse'], rmse)


def test_refit_nochange_reg(sim_nochange):
    """ Test refit ``keep_regularized=False`` (i.e., not ignoring coef == 0)
    """
    from sklearn.linear_model import LinearRegression as OLS
    estimator = OLS()

    refit = refit_record(sim_nochange, 'ols', estimator,
                         keep_regularized=False)
    assert 'ols_coef' in refit.dtype.names
    assert 'ols_rmse' in refit.dtype.names

    coef = np.array([[-3.83016528e+03, -3.83016528e+03],
                     [5.24635240e-03, 5.24635240e-03]])
    rmse = np.array([0.96794599, 0.96794599])
    np.testing.assert_allclose(refit[0]['ols_coef'], coef)
    np.testing.assert_allclose(refit[0]['ols_rmse'], rmse)


def test_refit_none():
    """ Test refit if model is None/[]
    """
    refit = refit_record(None, 'ols', None)
    assert refit is None
    refit = refit_record([], 'ols', None)
    assert refit is None


# ISSUE #79
def test_refit_issue_79(sim_nochange):
    """ Issue 79: missing coverage for case when record['coef'] are all zero

    Fix is to use ``refit_[(coef|rmse)]`` prefix variable to index the record
    name
    """
    from yatsm.regression import RLM
    estimator = RLM(maxiter=10)

    # Set record.coef to 0.
    sim_nochange.record['coef'] = np.zeros_like(sim_nochange.record['coef'])

    refit = refit_record(sim_nochange, 'rlm', estimator,
                         keep_regularized=True)
    assert 'rlm_coef' in refit.dtype.names
    assert 'rlm_rmse' in refit.dtype.names

    coef = np.zeros_like(sim_nochange.record[0]['coef'])
    rmse = np.array([0.97117668, 0.97117668])
    np.testing.assert_allclose(refit[0]['rlm_coef'], coef)
    np.testing.assert_allclose(refit[0]['rlm_rmse'], rmse)
