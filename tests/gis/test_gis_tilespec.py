""" Tests for `yatsm.gis.tilespec`
"""
import pytest

from yatsm.gis import tilespec


@pytest.fixture
def example_spec(request):
    for k in tilespec.TILESPECS:
        return tilespec.TILESPECS[k]


EX_CRS_GEOG = (
    (0., 0.),
    'epsg:4326',
    (0.00025, 0.0025),
    (1., 1.),
    'geographic'
)
EX_CRS_ALBERS = (
    (-2565600., 3314800.),
    'epsg:5070',
    (30, 30),
    (250, 250),
    'albers_conus'
)
EX_CRS_UTM = (
    (653385., 4828815.),
    'epsg:32619',
    (30, 30),
    (5000, 5000),
    'utm19n'
)


tilespec_params = pytest.mark.parametrize(
    ('ul', 'crs', 'res', 'size', 'desc'),
    [EX_CRS_GEOG,
     EX_CRS_ALBERS,
     EX_CRS_UTM]
)


# tilezilla.tilespec.TileSpec
@tilespec_params
def test_tilespec(ul, crs, res, size, desc):
    tilespec.TileSpec(ul, crs, res, size, desc=desc)


# FAILURE: CRS PARSING PROBLEMS
@tilespec_params
def test_tilespec_fail_crs_1(ul, crs, res, size, desc):
    with pytest.raises(ValueError):
        tilespec.TileSpec(ul, 'not a crs', res, size, desc)


# FAILURE: INDEXING PROBLEMS
def test_tilespec_fail_1(example_spec):
    with pytest.raises(IndexError):
        example_spec[-1]


def test_tilespec_fail_2(example_spec):
    with pytest.raises(TypeError):
        example_spec[([0, 1], [0, 1])]
