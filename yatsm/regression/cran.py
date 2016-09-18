""" Regression or prediction methods from R
"""
import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

Rstats = importr('stats')


def CRAN_spline(x, y, spar=0.55):
    """ Return a prediction function for a smoothing spline from R

    Use `rpy2` package to fit a smoothing spline using "smooth.spline".

    Args:
        x (np.ndarray): independent variable
        y (np.ndarray): dependent variable
        spar (float): smoothing parameter

    Returns:
        callable: prediction function of smoothing spline that provides
            smoothed estimates of the dependent variable given an input
            independent variable array

    Example:
      Fit a smoothing spline for y ~ x and predict for days in year:

        .. code-block:: python

            pred_spl = CRAN_spline(x, y)
            y_smooth = pred_spl(np.arange(1, 366))

    """
    spl = Rstats.smooth_spline(x, y, spar=spar)

    return lambda _x: np.array(Rstats.predict_smooth_spline(spl, _x)[1])
