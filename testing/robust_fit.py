# -*- coding: utf-8 -*
# vim: set expandtab:ts=4
"""
Perform an iteratively re-weighted least squares 'robust regression'.

Reference:
    http://www.mathworks.com/help/stats/robustfit.html
"""

import numpy as np

def mad_sigma(resid, p, c=0.6745):
    """
    Returns Median-Absolute-Deviation (MAD) for residuals of rank p.

    Inputs:
        resid       residuals
        p           rank of X
        c           scale factor to get to ~standard normal 
                        (i.e. 1 / 0.75iCDF ~= 1.4826 = 1 / 0.6745)
    
    Returns:
        mad_s        'robust' variance estimate

    Reference:
        http://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    # Sort residuals
    resid_sorted = np.sort(resid)
    # Return median absolute deviation adjusted sigma
    return np.median(resid_sorted[max(1, p):-1]) / c

def bisquare(resid):
    """
    Returns weighting for each residual using bisquare weight function

    Inputs:
        resid       residuals to be weighted

    Returns:
        weight      weights for residuals
    
    Reference:
        http://www.weizmann.ac.il/matlab/toolbox/curvefit/ch_fitt5.html
    """
    # Weight where abs(resid) < 1; otherwise 0
    return (np.abs(resid) < 1) * (1 - resid ** 2) ** 2

def weight_fit(y, X, w, rank):
    """
    Apply a weighted OLS fit to data
   
    # TODO
    """
    # Get square root of weights
    sw = np.sqrt(w)
    # Apply weights
    yw = y * sw
    Xw = X * sw
    # Get coefficients & residuals 
    return np.linalg.lstsq(Xw, yw)[0]

def robust_fit(x, y, weight_fun=bisquare, tune=4.685):
    """
    Perform robust fitting regression via iteratively reweighted least squares
    according to weight function and tuning parameter
    """
    # Get dimensions of input independent variable
    [n, p] = x.shape
    # Add in intercept column #TODO check speed
    X = np.insert(x, 0, np.ones(n), axis=1)
#    X = np.ones((n, p + 2))
#    X[:, 1:-1] = x
    [Q, R] = np.linalg.qr


















