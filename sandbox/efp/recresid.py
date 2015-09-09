# -*- coding: utf-8 -*-
u""" Recursive residuals computation

Citations:
Brown, RL, J Durbin, and JM Evans. 1975. Techniques for Testing the Consistency
    of Regression Relationships over Time. Journal of the Royal Statistical
    Society. Series B (Methodological) 37 (2): 149-192.

George G. Judge, William E. Griffiths, R. Carter Hill, Helmut Lütkepohl,
    and Tsoung-Chao Lee. 1985. The theory and practice of econometrics.
    New York: Wiley.
    ISBN: 978-0-471-89530-5

"""
import numba as nb
import numpy as np


def recresid(X, y, span=None):
    """ Return recursive residuals for y ~ X

    For matrix :math:`X_j` (:math:`j x K`) of the first :math:`j` rows of
    design matrix with :math:`K` features, calculate recursive residuals from
    estimate of :math:`y_j` using :math:`\\boldsymbol{b}_j`, which is least
    squared estimate of :math:`\\boldsymbol{\\beta}` based on the first
    :math:`j` observations.

    From Judge et al, 1985, chapter 5.5.2b (equation 5.5.13) and Brown, Durbin,
    and Evans (1975) (equation 2):

    .. math::
        e_j^* = \\frac{y_j - \\boldsymbol{x}_j^{\prime}\\boldsymbol{b}_{j-1}}
                      {(1 + \\boldsymbol{x}_j^{\prime}S_{j-1}\\boldsymbol{x}_j)^{1/2}}
              = \\frac{y_j - \\boldsymbol{x}_j^{\prime}\\boldsymbol{b}_j}
                      {(1 - \\boldsymbol{x}_j^{\prime}S_j\\boldsymbol{x}_j)^{1/2}},

        j = K + 1, K + 2, \ldots, T

    A quick way of calculating :math:`\\boldsymbol{b}_j` and
    :math:`\\boldsymbol{S}_j` is using an update formula (equation 5.5.14 and
    5.5.15 in Judge et al; Equations 4 and 5 in Brown, Durbin, and Evans):

    .. math::
        b_j = b_{j-1} + \\frac{S_{j-1}\\boldsymbol{x}_j(y_j - \\boldsymbol{x}_j^{\prime}\\boldsymbol{b}_{j-1})}
                              {1 + \\boldsymbol{x}_j^{\prime}S_{j-1}x_j}

    .. math::
        S_j = S_{j-1} -
            \\frac{S_{j-1}\\boldsymbol{x}_j\\boldsymbol{x}_j^{\prime}S_{j-1}}
                  {1 + \\boldsymbol{x}_j^{\prime}S_{j-1}\\boldsymbol{x}_j}

    See `statsmodels` implementation, which this follows:
    https://github.com/statsmodels/statsmodels/blob/88e67f22226bb56f947f5438782e3854d279af79/statsmodels/sandbox/stats/diagnostic.py#L1101


    Args:
        X (np.ndarray): 2D (n_features x n_obs) design matrix
        y (np.ndarray): 1D independent variable
        span (int, optional): minimum number of observations for initial
            regression. If ``span`` is None, use the number of features in
            ``X``

    Returns:
        np.ndarray: array containing recursive residuals standardized by
            prediction errors

    """
    nobs, nvars = X.shape
    if span is None:
        span = nvars

    recresid = np.nan * np.zeros((nobs))
    recvar = np.nan * np.zeros((nobs))

    X0 = X[:span]
    y0 = y[:span]

    # Initial fit
    XTX_j = np.linalg.inv(np.dot(X0.T, X0))
    XTY = np.dot(X0.T, y0)
    beta = np.dot(XTX_j, XTY)

    yhat_j = np.dot(X[span - 1], beta)
    recresid[span - 1] = y[span - 1] - yhat_j
    recvar[span - 1] = 1 + np.dot(X[span - 1],
                                  np.dot(XTX_j, X[span - 1]))
    for j in range(span, nobs):
        x_j = X[j:j + 1, :]
        y_j = y[j]

        # Prediction with previous beta
        yhat_j = np.dot(x_j, beta)
        resid_j = y_j - yhat_j

        # Update
        XTXx_j = np.dot(XTX_j, x_j.T)
        f_t = 1 + np.dot(x_j, XTXx_j)
        XTX_j = XTX_j - np.dot(XTXx_j, XTXx_j.T) / f_t # eqn 5.5.15

        beta = beta + (XTXx_j * resid_j / f_t).ravel()  # eqn 5.5.14

        recresid[j] = resid_j
        recvar[j] = f_t

    recresid_scaled = recresid / np.sqrt(recvar)
    sigma2 = recresid_scaled[span:].var(ddof=1)

    return recresid_scaled / np.sqrt(sigma2)
