""" GIS utilities
"""
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from shapely.geometry import Polygon

from yatsm.gis.core import Georeference
from yatsm.gis.utils import (bounds_to_polygon,
                             share_crs,
                             window_coords)
from yatsm.gis.conventions import (georeference_variable,
                                   make_xarray_crs,
                                   make_xarray_coords)
from yatsm.gis.convert import GIS_TO_STR, STR_TO_GIS


__all__ = [
    'Affine',
    'BoundingBox',
    'CRS',
    'Polygon',
    'bounds_to_polygon',
    'share_crs',
    'window_coords',
    'georeference_variable',
    'make_xarray_crs',
    'make_xarray_coords',
    'GIS_TO_STR', 'STR_TO_GIS',
    'Georeference'
]
