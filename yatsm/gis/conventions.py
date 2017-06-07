""" Attempts to be as CF-complaint as required... so far
"""
from collections import OrderedDict

import numpy as np
import xarray as xr

from yatsm.gis import projections
from yatsm.gis.utils import crs2osr


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


CF_NC_ATTRS = OrderedDict((
    ('Conventions', 'CF-1.7, YATSM'),
))


def georeference_variable(var, crs, transform):
    """ Georeference an `xarray.Variable` (e.g., DataArray)

    Args:
        var (xarray.Variable): Variable
        crs (CRS): Coordinate reference system
        transform (Affine): Affine transform

    Returns:
        xarray.Variable: Georeferenced variable
    """
    var.attrs['grid_mapping'] = 'crs'
    var.attrs['proj4'] = crs.to_string()
    var.attrs['crs_wkt'] = crs.wkt
    var.attrs['transform'] = transform

    return var


def make_xarray_crs(crs, transform):
    """ Return an `xarray.DataArray` of CF-compliant CRS info

    Args:
        crs (CRS): CRS
        transform (Affine): Affine transform

    Returns:
        xarray.DataArray: "crs" variable holding CRS information
    """
    name = projections.crs_name(crs)
    code = projections.epsg_code(crs) or 0

    da = xr.DataArray(np.array(code, dtype=np.int32), name='crs')
    da.attrs['grid_mapping_name'] = name

    da.attrs.update(projections.crs_names(crs))
    da.attrs.update(projections.crs_parameters(crs))
    da.attrs.update(projections.ellipsoid_parameters(crs))

    # For GDAL in case CF doesn't work
    # http://www.gdal.org/frmt_netcdf.html
    da.attrs['spatial_ref'] = crs.wkt
    da.attrs['GeoTransform'] = transform.to_gdal()

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
        units = crs_osr.GetLinearUnitsName().lower()
        y_attrs, x_attrs = COORD_DEFS['y'], COORD_DEFS['x']
        y_attrs['units'], x_attrs['units'] = units, units

    y = xr.Variable(('y', ), y, attrs=y_attrs)
    x = xr.Variable(('x', ), x, attrs=x_attrs)

    return y, x
