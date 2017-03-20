""" Projection parameters

.. todo::

    I want to keep the interface using just `rasterio.crs.CRS` objects, but I
    need to convert so many times. Consider:

        1. caching conversion
        2. move to OOP wrapper around CRS that defines these things,
           perhaps use throughout project? I don't really want to muck...

"""
from collections import OrderedDict
import logging

from .utils import crs2osr
from yatsm import errors

logger = logging.getLogger(__name__)


PROJECTION_DEFS = {
    # TODO: support more...
    # TODO: does the order matter?
    'albers_conical_equal_area': (
        'false_easting',
        'false_northing',
        'latitude_of_projection_origin',
        'longitude_of_central_meridian',
        'standard_parallel',
    ),
    'transverse_mercator': (
        'false_easting',
        'false_northing',
        'latitude_of_projection_origin',
        'longitude_of_central_meridian',
        'scale_factor',
    ),
    'universal_transverse_mercator': (
        'utm_zone_number'
    ),
}

ELLIPSOID_DEFS = OrderedDict((
    ('semi_major_axis', 'GetSemiMajor'),
    ('semi_minor_axis', 'GetSemiMinor'),
    ('inverse_flattening', 'GetInvFlattening')
))


def _epsg_key(crs):
    # "PROJCS", "GEOGCS", "GEOGCS|UNIT", NULL
    if crs.is_geographic:
        return 'GEOGCS'
    elif crs.is_projected:
        return 'PROJCS'
    else:
        return None


def epsg_code(crs):
    """ Return EPSG string (e.g., "epsg:32619") from a :ref:`rasterio.crs.CRS`

    Uses `OSRGetAuthorityCode`

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns:
        int: EPSG Code
    """
    crs_osr = crs2osr(crs)
    key = _epsg_key(crs)
    code = crs_osr.GetAuthorityCode(key)
    return int(code) if code else code


def crs_name(crs):
    """ Return name of a CRS projection

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns
        str: Lowercase projection name (see keys of :ref:`PROJECTION_DEFS`)

    """
    crs_osr = crs2osr(crs)
    return crs_osr.GetAttrValue('PROJECTION').lower()


def crs_long_name(crs):
    """ Return name of a CRS / ellipsoid pair

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns
        str: Lowercase projection name (see keys of :ref:`PROJECTION_DEFS`)

    """
    crs_osr = crs2osr(crs)
    return crs_osr.GetAttrValue('PROJCS')


def crs_parameters(crs):
    """ Return projection parameters for a CRS

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns
        OrderedDict: CRS parameters and values

    Raise
        yatsm.errors.TODO: Raise if CRS isn't supported yet
    """
    name = crs_name(crs)
    osr_crs = crs2osr(crs)

    if name not in PROJECTION_DEFS:
        raise errors.TODO('Cannot handle "{0}" CRS types yet'.format(name))

    return OrderedDict(
        (parm, osr_crs.GetProjParm(parm))
        for parm in PROJECTION_DEFS[name]
    )


def ellipsoid_parameters(crs):
    """ Return ellipsoid parameters for a CRS

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns
        OrderedDict: Ellipsoid parameters and values

    """
    osr_crs = crs2osr(crs)
    return OrderedDict(
        (key, getattr(osr_crs, func)())
        for (key, func) in ELLIPSOID_DEFS.items()
    )
