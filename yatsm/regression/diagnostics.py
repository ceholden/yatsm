""" Regression diagnostics calculations

Includes:
    - rmse: calculate root mean squared error
"""
import numpy as np

from ..accel import try_jit


@try_jit(nopython=True)
def rmse(y, yhat):
    """ Calculate and return Root Mean Squared Error (RMSE)

    Args:
        y (np.ndarray): known values
        yhat (np.ndarray): predicted values

    Returns:
        float: Root Mean Squared Error
    """
    return ((y - yhat) ** 2).mean() ** 0.5
