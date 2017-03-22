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

# See:
# https://trac.osgeo.org/gdal/wiki/NetCDF_ProjectionTestingStatus
# http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/cf-conventions.html#appendix-grid-mappings


#: dict: Mapping between OSGEO <-> CF projection names
CF_PROJECTION_NAMES = {
    'Albers_Conic_Equal_Area': 'albers_conic_equal_area',
    'Azimuthal_Equidistant': 'azimuthal_equidistant',
    'Lambert_Azimuthal_Equal_Area': 'lambert_azimuthal_equal_area',
    # ...
    'Transverse_Mercator': 'transverse_mercator',
}


#: dict: Mapping between CF <-> OSGEO projection parameters for projections
CF_PROJECTION_DEFS = {
    'albers_conic_equal_area': (
        ('latitude_of_projection_origin', 'latitude_of_center'),
        ('longitude_of_central_meridian', 'longitude_of_center'),
        ('standard_parallel', ('standard_parallel_1', 'standard_parallel_2')),
        ('false_easting', 'false_easting'),
        ('false_northing', 'false_northing'),
    ),
    'transverse_mercator': (
        ('latitude_of_projection_origin', 'latitude_of_origin'),
        ('longitude_of_central_meridian', 'central_meridian'),
        ('scale_factor_at_central_meridian', 'scale_factor'),
        ('false_easting', 'false_easting'),
        ('false_northing', 'false_northing'),
    ),
    'universal_transverse_mercator': (
        ('utm_zone_number', 'utm_zone_number')
    ),
    'latitude_longitude': (),  # None
}

#: tuple: Mapping between CF <-> OSGEO projection attribute name definitions
CF_CRS_NAMES = (
    ('horizontal_datum_name', 'GEOGCS|DATUM'),
    ('reference_ellipsoid_name', 'GEOGCS|DATUM|SPHEROID'),
    ('towgs84', 'GEOGCS|DATUM|TOWGS84'),
    ('prime_meridian_name', 'GEOGCS|PRIMEM')
)

#: tuple: mapping from CF ellipsoid parameter to SpatialReference method calls
CF_ELLIPSOID_DEFS = (
    ('semi_major_axis', 'GetSemiMajor'),
    ('semi_minor_axis', 'GetSemiMinor'),
    ('inverse_flattening', 'GetInvFlattening')
)
# TODO: support more...
# TODO: proj4 mappings


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
        str: Lowercase projection name (see keys of :ref:`CF_PROJECTION_DEFS`)

    """
    crs_osr = crs2osr(crs)
    if crs.is_projected:
        name = crs_osr.GetAttrValue('PROJECTION')
        if name not in CF_PROJECTION_NAMES:
            if crs.is_valid:
                raise errors.TODO('Cannot handle "{0}" CRS types yet: {1}'
                                  .format(name, crs.to_string()))
            else:
                raise KeyError('Unsupported CRS "{0!r}"'.format(crs))
        return CF_PROJECTION_NAMES[name]
    else:
        return 'latitude_longitude'


def crs_long_name(crs):
    """ Return name of a CRS / ellipsoid pair

    Args:
        crs (rasterio.crs.CRS): CRS

    Returns
        str: Lowercase projection name (see keys of :ref:`CF_PROJECTION_DEFS`)

    """
    crs_osr = crs2osr(crs)
    if crs.is_projected:
        return crs_osr.GetAttrValue('PROJCS')
    elif crs.is_geographic:
        return crs_osr.GetAttrValue('GEOGCS')


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

    if name not in CF_PROJECTION_DEFS:  # TODO: proj4 support
        raise errors.TODO('Cannot handle "{0}" CRS types yet'.format(name))

    parms = OrderedDict()  # "parm" is eww, but ref to `GetProjParm`
    for cf_parm, osgeo_parm in CF_PROJECTION_DEFS[name]:
        if isinstance(osgeo_parm, (list, tuple)):
            parms[cf_parm] = tuple(osr_crs.GetProjParm(p) for p in osgeo_parm)
        else:
            parms[cf_parm] = osr_crs.GetProjParm(osgeo_parm)
    return parms


def crs_names(crs):
    """ Return CF-compliant attributes to prevent "unknown" CRS/Ellipse/Geoid
    Args:
        crs (rasterio.crs.CRS): CRS

    Returns
        OrderedDict: CF attributes
    """
    osr_crs = crs2osr(crs)
    attrs = OrderedDict()

    long_name = crs_long_name(crs)
    if crs.is_projected:
        attrs['projected_coordinate_system_name'] = long_name
    elif crs.is_geographic:
        attrs['geographic_coordinate_system_name'] = long_name

    for cf_parm, osgeo_parm in CF_CRS_NAMES:
        value = osr_crs.GetAttrValue(osgeo_parm)
        if value:  # only record if not None so we can serialize
            attrs[cf_parm] = value
    return attrs


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
        for (key, func) in CF_ELLIPSOID_DEFS
    )
