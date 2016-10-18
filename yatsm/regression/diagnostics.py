""" Regression diagnostics calculations

Includes:
    - rmse: calculate root mean squared error
"""
import numpy as np

from ..accel import try_jit


#@try_jit(nopython=True, nogil=True)
def std(x, ddof=1):
    """ JIT-able implementation of standard deviation with degress of freedom

    Args:
        x (np.ndarray): Array of values
        ddof (int): Delta degrees of freedom. The divisor used in calculations
            is ``N - ddof`` where ``N`` is the number of elements in ``x``.

    Returns:
        float: The standard deviation
    """
    n = x.shape[0]
    m = x.sum() / (n - ddof)
    return (((x - m) ** 2).sum() / (n - ddof)) ** 0.5


@try_jit(nopython=True, nogil=True)
def rmse(y, yhat):
    """ Calculate and return Root Mean Squared Error (RMSE)

    Args:
        y (np.ndarray): known values
        yhat (np.ndarray): predicted values

    Returns:
        float: Root Mean Squared Error
    """
    return ((y - yhat) ** 2).mean() ** 0.5
