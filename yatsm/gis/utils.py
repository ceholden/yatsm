""" Geospatial utility functions
"""
import numpy as np
from osgeo import osr
import shapely.geometry

osr.UseExceptions()


def crs2osr(crs):
    """ Return `osgeo.osr.SpatialReference` of a `rasterio.crs.CRS`
    """
    crs_osr = osr.SpatialReference()
    crs_osr.ImportFromWkt(crs.wkt)
    return crs_osr


def bounds_to_polygon(bounds):
    """ Returns Shapely polygon of bounds

    Args:
        bounds (iterable): bounds (left bottom right top)

    Returns:
        shapely.geometry.Polygon: polygon of bounds

    """
    return shapely.geometry.Polygon([
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
        (bounds[0], bounds[3])
    ])


def window_coords(window, transform):
    """ Return Y/X coordinates for a given window and transform

    Args:
        window (tuple): Window ((ymin, ymax), (xmin, xmax)) in pixel
            space
        transform (affine.Affine): Affine transform

    Returns:
        tuple: Y/X coordinates

    """
    x0, y0 = transform.xoff, transform.yoff
    nx, ny = window[1][1] - window[1][0], window[0][1] - window[0][0]
    dx, dy = transform.a, transform.e

    coord_x = np.linspace(start=x0, num=nx, stop=(x0 + (nx - 1) * dx))
    coord_y = np.linspace(start=y0, num=ny, stop=(y0 + (ny - 1) * dy))

    return (coord_y, coord_x)
