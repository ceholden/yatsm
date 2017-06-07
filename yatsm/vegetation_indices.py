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

    Citation:
        Huete,A., K. Didan, T. Miura, E.P. Rodriguez, X. Gao and
            L.G. Ferreira. 2002. Overview of the radiometric and biophysical
            performance of the MODIS vegetation indices. Remote Sensing of
            Environment 83:195–213.

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


def EVI2(red, nir, scale_factor=1.0):
    """ Return the Enhanced Vegetation Index 2 for a set of np.ndarrays

    EVI2 is calculated as:

    .. math::
        2.5 * \\frac{(NIR - RED)}{(NIR + C * RED + L)}

    where:
        - :math:`RED` is the red band
        - :math:`NIR` is the near infrared band
        - :math:`G = 2.5` is the gain factor
        - :math:`C = 2.4`
        - :math:`L = 1`

    Note: bands must be given in float datatype from [0, 1]. If not,
          please provide the appropriate `scale_factor` to multiply
          by. This correction will be applied to the `L` parameter.

    Citation:
        Jiang, Z., A.R. Huete, K. Didan, and R. Miura. 2008. Development of a
            two-band enhanced vegetation index without a blue band. Remote
            Sensing of the Environment 112(10):3833-3845.

    Args:
        red (np.ndarray): red band
        nir (np.ndarray): NIR band
        scale_factor (float): Scaling factor for data (e.g., `10000` for
            scaled by 10,000)

    Returns:
        np.ndarray: EVI2

    """
    G = 2.5
    C = 2.4
    L = 1.0 * scale_factor

    return G * (nir - red) / (nir + C * red + L)
