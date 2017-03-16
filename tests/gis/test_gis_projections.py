""" Tests for `yatsm.gis.projections`
"""
import pytest

from yatsm.gis import projections


params = {
    5070: {'datum': 'NAD83',
           'lat_0': '23',
           'lat_1': '29.5',
           'lat_2': '45.5',
           'lon_0': '-96',
           'proj': 'aea',
           'units': 'm',
           'x_0': '0',
           'y_0': '0'},
    32619: {'datum': 'WGS84', 'proj': 'utm', 'units': 'm', 'zone': '19'},
    4326: {'datum': 'WGS84', 'proj': 'longlat'},
    3857: {'a': '6378137',
           'b': '6378137',
           'k': '1.0',
           'lat_ts': '0.0',
           'lon_0': '0.0',
           'nadgrids': '@null',
           'proj': 'merc',
           'units': 'm',
           'x_0': '0.0',
           'y_0': '0'},
    6491: {'ellps': 'GRS80',
           'lat_0': '41',
           'lat_1': '42.68333333333333',
           'lat_2': '41.71666666666667',
           'lon_0': '-71.5',
           'proj': 'lcc',
           'units': 'm',
           'x_0': '200000',
           'y_0': '750000'}
}


@pytest.mark.parametrize(('code', 'params'), list(params.items()))
def test_crs_parameters(code, params):
    _params = projections.crs_parameters(code)
    assert _params == params


@pytest.mark.parametrize('code', [-9999, 0])
def test_crs_parameters_fail(code):
    with pytest.raises(ValueError) as exc:
        projections.crs_parameters(code)
    assert 'Cannot find EPSG code' in str(exc)
