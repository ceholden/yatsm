from __future__ import print_function

import numpy as np
from glmnet.glmnet import elastic_net


def get_test_data():
    import os
    if os.path.isfile('test_X.npy') and os.path.isfile('test_y.npy'):
        X = np.load('test_X.npy')
        y = np.load('test_y.npy')
        return (X, y)
    else:
        return (None, None)


class CCDCLasso(object):

    """ Custom Lasso ElasticNet based on GLMNET """

    def __init__(self, X, y, alpha, rho=1.0):
        # Store variables
        # ensure X is n x p where n >= p
        if X.shape[0] >= X.shape[1]:
            self.X = X
        else:
            self.X = X.T
        self.y = y
        self.n_pred = X.shape[1]

        self.alpha = alpha
        self.rho = rho
        self._coef = None
        self._intercept = None
        self.rsquared = None

    def fit(self, **kwargs):
        """
        Runs elastic_net, corrects for order of coefficeints, and stores output
        of regression
        """
        # Store intercept, coefficients, pointers to coef in correct order,
        #   Rsquared and error term
#        _, a0, ca, ia, nin, rsq, _, _, jerr \
        n_lambdas, a0, ca, ia, nin, rsq, lambdas, _, jerr \
            = elastic_net(self.X, self.y, self.rho, None, None, **kwargs)

        # Record intercept
        self._intercept = a0

        # Remove predictors equal to zero and remove +1 for Fortran array
        self._coef = ca[:nin]
        self._indices = ia[:nin] - 1

        # Rsquared value
        self.rsquared = rsq

        return self

    def coef(self):
        coef = np.zeros(self.n_pred)
        coef[self._indices] = self._coef
        coef[0] = self._intercept

        return coef

    def fittedvalues(self):
        return (self._intercept +
                np.dot(self.X[:, self._indices], self._coef))[:, 0]

    def resid(self):
        return self.y - self.fittedvalues()

    def rmse(self):
        return np.linalg.norm(self.resid()) / np.sqrt(self.X.shape[0])

    def predict(self, X):
        """ Return y-hat values for a given X

        self._intercept + np.dot(X[self._indices], self._coef)
        """
        return (self._intercept + np.dot(X[self._indices], self._coef))

    def __str__(self):
        n_non_zeros = (np.abs(self.coef()) != 0).sum()
        return('%s with %d non-zero coefficients (%.2f%%)\n' +
               ' * Intercept = %.7f, Lambda = %.7f\n' +
               ' * Training r^2: %.4f') % \
            (self.__class__.__name__, n_non_zeros,
             n_non_zeros / float(len(self.coef())) * 100,
             self.intercept[0], self.alpha, self.rsquared[0])
