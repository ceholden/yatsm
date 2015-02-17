import math

import numpy as np
from glmnet.elastic_net import ElasticNet, elastic_net


class GLMLasso(ElasticNet):
    """ LASSO model using GLMnet implemented through `glmnet-python`.
    Basically a wrapper around the ElasticNet model that produces
    the variables I'm interested in within the `fit` method.
    """
    def __init__(self, alpha=1.0):
        super(GLMLasso, self).__init__(alpha)

    def fit(self, X, y, lambdas=None):
        if lambdas is None:
            lambdas = [self.alpha]
        elif not isinstance(lambdas, (np.ndarray, list)):
            lambdas = [lambdas]

        n_lambdas, intercept_, coef_, ia, nin, rsquared_, lambdas, _, jerr = \
            elastic_net(X, y, 1, lambdas=lambdas)
        # elastic_net will fire exception instead
        # assert jerr == 0

        # LASSO returns coefs out of order... reorder them with `ia`
        self.coef_ = np.zeros(X.shape[1])
        self.coef_[ia[:nin[0]] - 1] = coef_[:nin[0], 0]

        self.intercept_ = intercept_
        self.rsquared_ = rsquared_

        # Create external friendly coefficients
        self.coef = np.copy(self.coef_)
        self.coef[0] += intercept_

        # Store number of observations
        self.nobs = y.size

        # Store fitted values
        self.fittedvalues = self.predict(X)

        # Calculate the residual sum of squares
        self.rss = np.sum((y - self.fittedvalues) ** 2)

        # Calculate model RMSE
        self.rmse = math.sqrt(self.rss / self.nobs)

        return self
