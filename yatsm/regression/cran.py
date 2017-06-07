""" Regression or prediction methods from R
"""
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

Rstats = importr('stats')


def CRAN_spline(x, y, w=None, **kwds):
    """ Return a prediction function for a smoothing spline from R

    Use `rpy2` package to fit a smoothing spline using "smooth.spline".

    Args:
        x (np.ndarray): independent variable
        y (np.ndarray): dependent variable
        w (np.ndarray): weights for each observation in `x`/`y`

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
    if w is not None:
        spl = Rstats.smooth_spline(x, y, w, **kwds)
    else:
        spl = Rstats.smooth_spline(x, y, **kwds)

    return lambda _x: np.array(Rstats.predict_smooth_spline(spl, _x)[1])
