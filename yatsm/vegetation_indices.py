""" Functions for computing vegetation indices
"""
from __future__ import division


def EVI(red, nir, blue):
    """ Return the Enhanced Vegetation Index for a set of np.ndarrays

    EVI is calculated as:

    .. math::
        2.5 * \\frac{(NIR - RED)}{(NIR + C_1 * RED - C_2 * BLUE + L)}

    where:
        - :math:`RED` is the red band
        - :math:`NIR` is the near infrared band
        - :math:`BLUE` is the blue band
        - :math:`C_1 = 6`
        - :math:`C_2 = 7.5`
        - :math:`L = 1`

    Note: bands must be given in float datatype from [0, 1]

    Args:
      red (np.ndarray): red band
      nir (np.ndarray): NIR band
      blue (np.ndarray): blue band

    Returns:
      np.ndarray: EVI

    """
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

def NDVI(red, nir):
    """ Return the Normalized Difference Vegetation Index for a set of np.ndarrays

    NDVI is calculated as:

    .. math::
        \\frac{(NIR - RED)}{(NIR + RED )}

    where:
        - :math:`RED` is the red band
        - :math:`NIR` is the near infrared band

    Note: bands must be given in float datatype from [0, 1]

    Args:
      red (np.ndarray): red band
      nir (np.ndarray): NIR band

    Returns:
      np.ndarray: NDVI

    """
    return (nir - red) / (nir + red)
