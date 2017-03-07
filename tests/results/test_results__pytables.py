""" Tests for ``yatsm.results._pytables``
"""
import shapely.wkt
import pytest
import tables as tb

from yatsm.gis import Affine, CRS, BoundingBox, Georeference
from yatsm.results import HDF5ResultsStore

# Fixtures and definitions
_CRS = CRS({'init': 'epsg:32619'})
_BOUNDS = BoundingBox(0, 0, 10, 10)
_TRANSFORM = Affine(0.2,  0.0, -114,
                    0.0, -0.2, 46)
_BBOX = shapely.wkt.loads('POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))')

_GEOREF = Georeference(_CRS, _BOUNDS, _TRANSFORM, _BBOX)


@pytest.fixture(scope='function')
def test_data_1(tmpdir):
    with HDF5ResultsStore(str(tmpdir.join('1.nc')), georef=_GEOREF) as store:
        return store


class TestHDF5ResultsStore(object):

    # CREATION
    def test_create_no_georef(self, tmpdir):
        with pytest.raises(TypeError) as te:
            HDF5ResultsStore(str(tmpdir.join('1.nc')), georef=None)
        assert 'Must specify `georef` as `Georeference`' in str(te.value)

    # CONTEXT MANAGER
    def test_with_write(self):
        pass

    def test_with_modify(self):
        pass

    def test_with_read(self):
        pass

    # GIS METADATA
    def test_georeference(self, test_data_1):
        assert test_data_1.crs == _CRS

    def test_bounds(self, test_data_1):
        assert test_data_1.bounds == _BOUNDS

    def test_transform(self, test_data_1):
        assert test_data_1.transform == _TRANSFORM

    def test_bbox(self, test_data_1):
        assert test_data_1.bbox == _BBOX

    # METADATA
    def test_basename(self, test_data_1):
        assert test_data_1.basename == '1.nc'

    def test_tags(self, test_data_1):
        assert all([tagname in test_data_1.tags for tagname in
                    Georeference._fields])

    def test_write_tags(self, test_data_1):
        tags = {
            'much': 'so',
            'data': 'wow'
        }
        test_data_1.close()
        with HDF5ResultsStore(test_data_1.filename, 'r+') as h5:
            h5.update_tags(**tags)

        for tagname, value in tags.items():
            assert tagname in h5.tags
            assert h5.tags[tagname] == value

    # DICT
    def test_keys(self, test_data_1):
        assert list(test_data_1.keys()) == ['/']

    def test_items(self, test_data_1):
        items = dict(test_data_1.items())
        assert '/' in items
        assert isinstance(items['/'], tb.Group)

    def test_getitem(self):
        pass

    def test_setitem(self):
        pass
