""" Functions for computing vegetation indices
"""
from __future__ import division


def EVI(red, nir, blue, scale_factor=1.0):
    """ Return the Enhanced Vegetation Index for a set of np.ndarrays

    EVI is calculated as:

    .. math::
        G * \\frac{(NIR - RED)}{(NIR + C_1 * RED - C_2 * BLUE + L)}

    where:
        - :math:`RED` is the red band
        - :math:`NIR` is the near infrared band
        - :math:`BLUE` is the blue band
        - :math:`G = 2.5` is the gain factor
        - :math:`C_1 = 6`
        - :math:`C_2 = 7.5`
        - :math:`L = 1`

    Note: bands must be given in float datatype from [0, 1]. If not,
          please provide the appropriate `scale_factor` to multiply
          by. This correction will be applied to the `L` parameter.

    Args:
        red (np.ndarray): red band
        nir (np.ndarray): NIR band
        blue (np.ndarray): blue band
        scale_factor (float): Scaling factor for data (e.g., `10000` for
            scaled by 10,000)

    Returns:
        np.ndarray: EVI

    """
    G = 2.5
    C_1 = 6
    C_2 = 7.5
    L = 1.0 * scale_factor

    return G * (nir - red) / (nir + C_1 * red - C_2 * blue + L)
