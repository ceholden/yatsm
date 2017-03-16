""" Attempts to be as CF-complaint as required... so far
"""
import numpy as np
import xarray as xr

from yatsm.gis.utils import crs2osr


PROJ_DEFS = {
    # TODO: support more...
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
ELLIPSOID_DEFS = (
    'semi_major_axis',
    'semi_minor_axis',
    'inverse_flattening'
)

COORD_DEFS = {
    'longitude': {
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east',
    },
    'latitude': {
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north',
    },
    'x': {
        'standard_name': 'projection_x_coordinate',
        'long_name': 'x coordinate of projection',
    },
    'y': {
        'standard_name': 'projection_y_coordinate',
        'long_name': 'y coordinate of projection',
    },
    # TODO: currently nothing is using the right time stamps (usually ordinal),
    #       but this is what it should be
    'time': {
        'standard_name': 'time',
        'long_name': 'Time, unix time-stamp',
        'axis': 'T',
        'calendar': 'standard'
    }
}


def make_xarray_crs(crs):
    crs_osr = crs2osr(crs)

    name = crs_osr.GetAttrValue('PROJECTION').lower()
    proj_params_list = PROJ_DEFS[name]

    da = xr.DataArray(np.array([0], dtype=np.int32), name='crs')
    da.attrs['grid_mapping_name'] = name
    da.attrs['long_name'] = crs_osr.GetAttrValue('PROJCS')
    for param in proj_params_list:
        da.attrs[param] = crs_osr.GetProjParm(param)
    for param in ELLIPSOID_DEFS:
        da.attrs[param] = crs_osr.GetAttrValue(param)

    from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA

    return da


def make_xarray_coords(y, x, crs):
    """ Return `y` and `x` as `xr.Variable` useful for coordinates

    Args:
        y (np.ndarray): Y
        x (np.ndarray): X
        crs (CRS): Coordinate reference system

    Returns:
        tuple: Y and X
    """
    if crs.is_geographic:
        y_attrs, x_attrs = COORD_DEFS['latitude'], COORD_DEFS['longitude']
    elif crs.is_projected:
        crs_osr = crs2osr(crs)
        units = crs_osr.GetLinearUnitsName()
        y_attrs, x_attrs = COORD_DEFS['y'], COORD_DEFS['x']
        y_attrs['units'], x_attrs['units'] = units, units

    y = xr.Variable(('y', ), y, attrs=y_attrs)
    x = xr.Variable(('x', ), x, attrs=x_attrs)

    return y, x
