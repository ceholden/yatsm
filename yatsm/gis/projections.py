""" Projection parameters
"""
import logging
from pathlib import Path
import pyproj
import re

from .utils import crs2osr

logger = logging.getLogger(__name__)


def epsg_code(crs):
    """ Try to find EPSG code from a :ref:`rasterio.crs.CRS`

    Uses `OSRGetAuthorityName` and `OSRGetAuthorityCode`

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns:
        str: [EPSG Authority]:[EPSG Code]
    """
    crs_osr = crs2osr(crs)
    # "PROJCS", "GEOGCS", "GEOGCS|UNIT", NULL
    if crs.is_geographic:
        key = 'GEOGCS'
    elif crs.is_projected:
        key = 'PROJCS'
    else:
        key = None
    return '{0}:{1}'.format(crs_osr.GetAuthorityName(key),
                            crs_osr.self.GetAuthorityCode(key)).lower()


def crs_parameters(epsg_code):
    """ Return projection parameters for a projection denoted by an EPSG code

    Args:
        epsg_code (int): EPSG code for a projection

    Returns
        dict: Mapping projection names (str) to projection parameters

    Raises
        ValueError: Raise if EPSG code is not found in ``pyproj`` data file
    """

    # Proj.4 parameters stored in 'epsg' data file
    # Example:
    # WGS 84 / UTM zone 19N
    # <32619> +proj=utm +zone=19 +datum=WGS84 +units=m +no_defs  <>
    epsg_data = Path(pyproj.pyproj_datadir).joinpath('epsg')
    with open(str(epsg_data)) as f:
        content = f.read()

    match = re.search('(?<=<{0}>).*(?=<>)'.format(epsg_code), content)
    if not match:
        raise ValueError("Cannot find EPSG code {0} in pyproj data file"
                         .format(epsg_code))
    parameters = match.group().strip()

    parameters = dict(attr.split('=') for attr in
                      parameters.replace('+', '').split(' ') if '=' in attr)
    return parameters
