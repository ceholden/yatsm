""" Tests for ``yatsm.gis.convert``
"""
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import pytest

from yatsm.gis import convert


@pytest.mark.parametrize(('numbers', 'string'), (
    ((1, 2, 3, 4, ),
     'BoundingBox(left=1, bottom=2, right=3, top=4)'),
    ((1.0, 2.0, 3.0, 4.0, ),
     'BoundingBox(left=1.0, bottom=2.0, right=3.0, top=4.0)'),
    ((-1.0, 2, 3.0, -4.0, ),
     'BoundingBox(left=-1.0, bottom=2, right=3.0, top=-4.0)'),

))
def test_convert_bounds(numbers, string):
    b = BoundingBox(*numbers)
    assert convert.bounds_to_str(b) == string
    assert convert.str_to_bounds(string) == b


@pytest.mark.parametrize(('numbers', 'string'), (
    ((1, 2, 3, 4, 5, 6, ),
     'Affine(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)'),
    ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, ),
     'Affine(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)'),
))
def test_convert_transform(numbers, string):
    test = Affine(*numbers)
    assert convert.GIS_TO_STR['transform'](test) == string
    assert convert.STR_TO_GIS['transform'](string) == test


@pytest.mark.parametrize('string', (
    '+init=EPSG:5070',
    '+init=EPSG:32619'
))
def test_convert_crs(string):
    test = CRS.from_string(string)
    assert convert.GIS_TO_STR['crs'](test) == string
    assert convert.STR_TO_GIS['crs'](string) == test
