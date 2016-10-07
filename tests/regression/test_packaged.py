""" Tests for finding pre-packaged regressors
"""
import os

import pytest
import sklearn.externals.joblib as jl

from yatsm.regression import packaged


@pytest.mark.parametrize('name', packaged.packaged_regressions,
                         ids=packaged.packaged_regressions)
def test_packaged_regressors(name):
    fname = packaged.find_packaged_regressor(name)
    assert os.path.isfile(fname)
    jl.load(fname)


@pytest.mark.parametrize('name', ['asdf', 'foobar'])
def test_unknown_regressor(name):
    with pytest.raises(KeyError):
        packaged.find_packaged_regressor(name)
