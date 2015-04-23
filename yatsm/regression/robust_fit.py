"""
Perform an iteratively re-weighted least squares 'robust regression'. Basically
a clone of `statsmodels.robust.robust_linear_model.RLM` without all the lovely,
but costly, creature comforts.

Reference:
    http://statsmodels.sourceforge.net/stable/rlm.html
    http://cran.r-project.org/web/packages/robustreg/index.html
    http://cran.r-project.org/doc/contrib/Fox-Companion/appendix-robust-regression.pdf

"""

import numpy as np
import scipy.linalg


# Weight scaling methods
def bisquare(resid, c=4.685):
    """
    Returns weighting for each residual using bisquare weight function

    Args:
      resid (np.ndarray): residuals to be weighted
      c (float): tuning constant for Tukey's Biweight (default: 4.685)

    Returns:
        weight (ndarray): weights for residuals

    Reference:
        http://statsmodels.sourceforge.net/stable/generated/statsmodels.robust.norms.TukeyBiweight.html
    """
    # Weight where abs(resid) < c; otherwise 0
    return (np.abs(resid) < c) * (1 - (resid / c) ** 2) ** 2


def mad(resid, c=0.6745):
    """
    Returns Median-Absolute-Deviation (MAD) for residuals

    Args:
      resid (np.ndarray): residuals
      c (float): scale factor to get to ~standard normal (default: 0.6745)
                 (i.e. 1 / 0.75iCDF ~= 1.4826 = 1 / 0.6745)

    Returns:
      float: MAD 'robust' variance estimate

    Reference:
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    # Return median absolute deviation adjusted sigma
    return np.median(np.fabs(resid)) / c


# Utility functions
def _check_converge(x0, x, tol=1e-8):
    return not np.any(np.fabs(x0 - x > tol))


def _weight_fit(y, X, w):
    """
    Apply a weighted OLS fit to data

    Args:
      y (ndarray): dependent variable
      X (ndarray): independent variables
      w (ndarray): observation weights

    Returns:
      tuple: coefficients, residual vector, and RSS

    """
    sw = np.sqrt(w)

    Xw = X * sw[:, None]
    yw = y * sw

    beta, _, _, _ = np.linalg.lstsq(Xw, yw)

    resid = y - np.dot(X, beta)
    rss = (resid ** 2).sum()

    return beta, resid, rss


# Robust regression
class RLM(object):
    """
    Perform robust fitting regression via iteratively reweighted least squares
    according to weight function and tuning parameter.

    Basically a clone from `statsmodels` that should be much faster.

    Args:
      y (np.ndarray): dependent variable vector
      X (np.ndarray): independent variable matrix
      scale_est (callable): function for scaling residuals
      tune (float): tuning constant for scale estimate

    Attributes:
      y (np.ndarray): dependent variable
      X (np.ndarray): independent variable design matrix
      M (callable): function for scaling residuals
      tune (float): tuning constant for scale estimate
      coefs (np.ndarray): coefficients of fitted model
      resid (np.ndarray): residuals of the fitted models (y - fitted values)
      rss (float): Residual Sum of Squares (RSS) of model

    """

    def __init__(self, y, X, M=bisquare, tune=4.685):
        self.y = y
        self.X = X
        self.M = M
        self.tune = tune

        if self.X.ndim == 1:
            self.X = self.X[:, np.newaxis]

        self.weights = np.ones_like(y)

    def fit(self, maxiter=50, tol=1e-8, scale_est=mad, scale_constant=0.6745,
            update_scale=True):
        """ Fit model using iteratively reweighted least squares

        Args:
          maxiter (int, optional): maximum number of iterations (default: 50)
          tol (float, optional): convergence tolerance of estimate (default:
            1e-8)
          scale_est (callable): estimate used to scale the weights (default:
            `mad` for median absolute deviation)
          scale_constant (float): normalization constant (default: 0.6745)
          update_scale (bool, optional): update scale estimate for weights
            across iterations (default: True)

        Returns:
          self

        """
        self.scale_est = scale_est

        self.coefs, self.resid, self.rss = \
            _weight_fit(self.y, self.X, self.weights)
        self.scale = self.scale_est(self.resid, c=scale_constant)

        iteration = 1
        converged = 0
        while not converged and iteration < maxiter:
            _coefs = self.coefs.copy()
            self.weights = self.M(self.resid / self.scale, c=self.tune)
            self.coefs, self.resid, self.rss = \
                _weight_fit(self.y, self.X, self.weights)
            if update_scale is True:
                self.scale = self.scale_est(self.resid, c=scale_constant)
            iteration += 1
            converged = _check_converge(self.coefs, _coefs, tol=tol)

        return self

    def predict(self, X=None):
        """ Return linear model prediction after it has been fitted

        Args:
          X (np.ndarray, optional): user provided design matrix

        Returns:
          np.ndarray: yhat model predictions

        """
        if X is None:
            X = self.X
        return np.dot(self.X, self.coefs)


if __name__ == '__main__':
    # Do some tests versus `statsmodels.robust.RLM`
    import timeit

    setup = 'from __main__ import RLM; import statsmodels.api as sm; import numpy as np; np.random.seed(123456789); y = np.random.rand(1000); X = np.random.rand(1000, 4)'
    my_rlm = 'my_rlm = RLM(y, X).fit()'
    sm_rlm = 'sm_rlm = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight()).fit(conv="coefs")'

    ns = {}
    exec setup in ns
    exec my_rlm in ns
    exec sm_rlm in ns
    if np.allclose(ns['my_rlm'].coefs, ns['sm_rlm'].params):
        print('Pass: Two RLM solutions produce the same answers')
    else:
        print('Error: Two RLM solutions do not produce the same answers')

    t_my_rlm = timeit.timeit(stmt=my_rlm, setup=setup, number=1000)
    t_sm_rlm = timeit.timeit(stmt=sm_rlm, setup=setup, number=1000)

    print('My RLM: {t}s'.format(t=t_my_rlm))
    print('statsmodels RLM: {t}s'.format(t=t_sm_rlm))
    print('Speedup: {t}%'.format(t=t_sm_rlm / t_my_rlm * 100))
