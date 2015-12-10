""" Setup script to pickle various statistical estimators for distribution

Available pickles to build:
    * glmnet_Lasso20.pkl
    * sklearn_Lasso20.pkl

"""
from __future__ import print_function

import json
import logging
import os
import traceback

# Don't alias to ``np``: https://github.com/numba/numba/issues/1559
import numpy
import sklearn.linear_model
from sklearn.externals import joblib as jl
import six

logger = logging.getLogger()


# GLMNET pickles
try:
    import glmnet
    _glmnet_pickles = {
        'glmnet_Lasso20.pkl': glmnet.Lasso(lambdas=20),
        'glmnet_LassoCV_n50.pkl': glmnet.LassoCV(
            lambdas=numpy.logspace(1e-4, 35, 50)),
    }
except:
    logger.error('Could not produce pickles from package "glmnet". '
                 'Check if it is installed')
    print(traceback.format_exc())
    _glmnet_pickles = {}

# scikit-learn pickles
_sklearn_pickles = {
    'OLS.pkl': sklearn.linear_model.LinearRegression(),
    'sklearn_Lasso20.pkl': sklearn.linear_model.Lasso(alpha=20.0),
    'sklearn_LassoCV_n50.pkl': sklearn.linear_model.LassoCV(
        alphas=numpy.logspace(1e-4, 35, 50)),
}

# YATSM pickles
from ..robust_fit import RLM  # flake8: noqa
_yatsm_pickles = {
    'rlm_maxiter10.pkl': RLM(maxiter=10)
}

pickles = [_glmnet_pickles, _sklearn_pickles, _yatsm_pickles]
here = os.path.dirname(__file__)
pickles_json = os.path.join(here, 'pickles.json')


def make_pickles():
    logger.info('Serializing estimators to pickles...')
    packaged = {}

    for pickle in pickles:
        for fname, obj in six.iteritems(pickle):
            jl.dump(obj, os.path.join(here, fname), compress=5)
            packaged[os.path.splitext(fname)[0]] = obj.__class__.__name__

    with open(pickles_json, 'w') as f:
        json.dump(packaged, f, indent=4)
        logger.info('Wrote pickles.json to %s' % pickles_json)
