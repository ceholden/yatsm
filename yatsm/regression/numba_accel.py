""" Regression related functions optimized using Numba, if available

If Numba is not avaiable, functions are still usable as not JIT'd functions
"""


# @nb.jit()
def rmse(y, yhat):
    """ Calculate and return Root Mean Squared Error (RMSE)

    Args:
        y (np.ndarray): known values
        yhat (np.ndarray): predicted values

    Returns:
        float: Root Mean Squared Error
    """
    return ((y - yhat) ** 2).mean() ** 0.5


try:
    import numba as nb
except:
    pass
else:
    # JIT functions if we can
    rmse = nb.jit(rmse)
