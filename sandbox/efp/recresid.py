# -*- coding: utf-8 -*-
u""" Recursive residuals computation

Citations:
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

    From Judge et al, 1985, chapter 5.5.2b (equation 5.5.13):

    .. math::
        e_j^* = \\frac{y_j - \\boldsymbol{x}_j^{\prime}\\boldsymbol{b}_{j-1}}
                      {(1 + \\boldsymbol{x}_j^{\prime}S_{j-1}\\boldsymbol{x}_j)^{1/2}}
              = \\frac{y_j - \\boldsymbol{x}_j^{\prime}\\boldsymbol{b}_j}
                      {(1 - \\boldsymbol{x}_j^{\prime}S_j\\boldsymbol{x}_j)^{1/2}},

        j = K + 1, K + 2, \ldots, T

    A quick way of calculating :math:`\\boldsymbol{b}_j` and
    :math:`\\boldsymbol{S}_j` is using an update formula (equation 5.5.14 and
    5.5.15):

    .. math::
        b_j = b_{j-1} + \\frac{S_{j-1}\\boldsymbol{x}_j(y_j - \\boldsymbol{x}_j^{\prime}\\boldsymbol{b}_{j-1})}
                              {1 + \\boldsymbol{x}_j^{\prime}S_{j-1}x_j}

    .. math::
        S_j = S_{j-1} -
            \\frac{S_{j-1}\\boldsymbol{x}_j\\boldsymbol{x}_j^{\prime}S_{j-1}}
                  {1 + \\boldsymbol{x}_j^{\prime}S_{j-1}\\boldsymbol{x}_j}


    Args:
        X (np.ndarray): 2D (n_features x n_obs) design matrix
        y (np.ndarray): 1D independent variable
        span (int, optional): minimum number of observations for initial
            regression. If ``span`` is None, use the number of features in
            ``X``

    Returns:
        np.ndarray: recursive residuals

    """
    pass
