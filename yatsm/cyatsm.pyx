import numpy as np
import statsmodels.api as sm
cimport numpy as np
cimport cython

cdef int green_band = 1
cdef int swir1_band = 4
cdef double ndays = 365.25


def multitemp_mask(np.ndarray x, np.ndarray Y, double n_year,
                   double crit=400,
                   int green=green_band, int swir1=swir1_band,
                   int maxiter=10):
    """ Multi-temporal masking using RLM

    Taken directly from CCDC (Zhu and Woodcock, 2014). This "temporal masking"
    procedure was ported from CCDC v9.3.

    Args:
      x (ndarray): array of ordinal dates
      Y (ndarray): matrix of observed spectra
      n_year (float): "number of years to mask"
      crit (float, optional): critical value for masking clouds/shadows
      green (int, optional): 0 indexed value for green band in Y
      swir1 (int, optional): 0 indexed value for SWIR (~1.55-1.75um) band in Y
      maxiter (int, optional): maximum iterations for RLM fit

    Returns:
      mask (ndarray): mask where False indicates values to be masked

    """
    n_year = np.ceil(n_year)

    cdef double w = 2.0 * np.pi / ndays

    cdef np.ndarray X = np.array([
        np.ones_like(x),
        np.cos(w * x),
        np.sin(w * x),
        np.cos(w / n_year * x),
        np.sin(w / n_year * x)
    ])

    green_RLM = sm.RLM(Y[green, :], X.T,
                       M=sm.robust.norms.TukeyBiweight())
    swir1_RLM = sm.RLM(Y[swir1, :], X.T,
                       M=sm.robust.norms.TukeyBiweight())

    return np.logical_or(green_RLM.fit(maxiter=maxiter).resid < crit,
                         swir1_RLM.fit(maxiter=maxiter).resid > -crit)
