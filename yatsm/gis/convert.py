""" GIS datatypes to/from string
"""
import re

from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import shapely.geometry
import shapely.wkt

from yatsm.utils import to_number


RE_NUMBER = re.compile(r'[-+]?\d*\.\d+|\d+')


def bounds_to_str(bounds):
    """ Return string repr of BoundingBox
    """
    return repr(bounds)


def str_to_bounds(string):
    return BoundingBox(*map(to_number, RE_NUMBER.findall(string)))


def transform_to_str(transform):
    """ Return `str` representation of `ref`:Affine:
    """
    return 'Affine({0})'.format(
        ', '.join([str(getattr(transform, member))
                   for member in 'abcdef']))


def str_to_transform(string):
    """ Return :class:`Affine` from a `str`
    """
    numbers = RE_NUMBER.findall(string)
    if len(numbers) != 6:
        if 'affine' in string.lower():
            raise ValueError('Transform expects 6 values')
        else:
            raise ValueError('Cannot decipher transform "{0}"'
                             .format(string))
    return Affine(*map(float, numbers))


def bbox_to_str(bbox):
    return bbox.to_wkt()


def str_to_bbox(string):
    return shapely.wkt.loads(string)


GIS_TO_STR = {
    'crs': lambda crs: CRS(crs).to_string(),
    'bounds': bounds_to_str,
    'transform': transform_to_str,
    'bbox': bbox_to_str
}


STR_TO_GIS = {
    'crs': lambda s: CRS.from_string(s),
    'bounds': str_to_bounds,
    'transform': str_to_transform,
    'bbox': str_to_bbox
}
