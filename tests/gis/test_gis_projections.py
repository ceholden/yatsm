""" Tests for `yatsm.gis.projections`

For info on projection definitions in various systems (proj4, osgeo, CF), see:

    https://trac.osgeo.org/gdal/wiki/NetCDF_ProjectionTestingStatus
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.7/build/cf-conventions.html#appendix-grid-mappings

"""
import pytest

from yatsm.gis import CRS, projections


CF_PARAMS = {
    5070: {
        # ellipsoid_parameters
        'ELLIPSOID': {
            'semi_major_axis': 6378137.0,
            'semi_minor_axis': 6356752.314140356,
            'inverse_flattening': 298.257222101
        },
        # crs_names
        'NAMES': {
            'horizontal_datum_name': 'North_American_Datum_1983',
            'reference_ellipsoid_name': 'GRS 1980',
            'towgs84': '0',
            'prime_meridian_name': 'Greenwich'
        },
        # crs_parameters
        'CRS': {
            'latitude_of_projection_origin': 23,
            'standard_parallel': (29.5, 45.5),
            'longitude_of_central_meridian': -96,
            'false_easting': 0,
            'false_northing': 0
        }
    },
    32619: {
        # ellipsoid_parameters
        'ELLIPSOID': {
            'semi_major_axis': 6378137.0,
            'semi_minor_axis': 6356752.314245179,
            'inverse_flattening': 298.257223563
        },
        # crs_names
        'NAMES': {
            'horizontal_datum_name': 'WGS_1984',
            'reference_ellipsoid_name': 'WGS 84',
            'prime_meridian_name': 'Greenwich',
        },
        # crs_parameters
        'CRS': {
            'latitude_of_projection_origin': 0.0,
            'longitude_of_central_meridian': -69.0,
            'scale_factor_at_central_meridian': 0.9996,
            'false_easting': 500000.0,
            'false_northing': 0.0,
        }
    },
    4326: {
        'ELLIPSOID': {
            'semi_major_axis': 6378137,
            'semi_minor_axis': 6356752.314245179,
            'inverse_flattening': 298.257223563,
        },
        'NAMES': {
            'horizontal_datum_name': 'WGS_1984',
            'reference_ellipsoid_name': 'WGS 84',
            'prime_meridian_name': 'Greenwich',
        },
        'CRS': {},  # None
    },
}
#    6491: {
#       'reference_ellipsoid_name': 'GRS80',
#       'latitude_of_projection_origin': '41',
#       'standard_parallel': ('42.68333333333333', '41.71666666666667'),
#       'longitude_of_central_meridian': '-71.5',
#       'proj': 'lcc',
#       'units': 'm',
#       'false_easting': '200000',
#       'false_northing': '750000'
#    }
#    3857: {
#        'ELLIPSOID': {
#            'semi_major_axis': 6378137.0,
#            'semi_minor_axis': 6356752.314245179,
#            'inverse_flattening': 298.257223563
#        },
#        # crs_names
#        'NAMES': {
#            'horizontal_datum_name': 'WGS_1984',
#            'reference_ellipsoid_name': 'WGS 84',
#            'towgs84': None,
#            'prime_meridian_name': 'Greenwich',
#        },
#        'CRS': {
#            # TODO
#        },
#    },


def _check_dict(test, true):
    for k, v in true.items():
        assert (test[k] == v or v is None)


@pytest.mark.parametrize(('code', 'params'), list(CF_PARAMS.items()))
def test_crs_names_CF(code, params):
    crs = CRS.from_epsg(code)
    crs_names = projections.crs_names(crs)

    _check_dict(crs_names, params['NAMES'])


@pytest.mark.parametrize(('code', 'params'), list(CF_PARAMS.items()))
def test_crs_parameters_CF(code, params):
    crs = CRS.from_epsg(code)

    crs_params = projections.crs_parameters(crs)
    _check_dict(crs_params, params['CRS'])


@pytest.mark.parametrize(('code', 'params'), list(CF_PARAMS.items()))
def test_ellipsoid_parameters_CF(code, params):
    crs = CRS.from_epsg(code)
    ellips_params = projections.ellipsoid_parameters(crs)

    _check_dict(ellips_params, params['ELLIPSOID'])


@pytest.mark.parametrize('code', [3857])
def test_crs_parameters_TODO(code):
    crs = CRS.from_epsg(code)
    with pytest.raises(NotImplementedError) as exc:
        projections.crs_parameters(crs)
    assert 'TODO' in str(exc)
