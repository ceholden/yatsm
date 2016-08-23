""" Tests for yatsm.classification.roi
"""
import os

import fiona
import numpy as np
import pytest
import rasterio
import shapely.geometry

from yatsm.classification import roi

DATA = os.path.join(os.path.dirname(__file__), 'data')
TRAINING_ROI = os.path.join(DATA, 'training.geojson')
TRAINING_RASTER = os.path.join(DATA, 'training.gtif')


@pytest.fixture()
def training_roi():
    with fiona.open(TRAINING_ROI) as ds:
        training = list(ds)
    return training


@pytest.fixture
def training_raster():
    # dataset is 1 band and all values == 255
    ds = rasterio.open(TRAINING_RASTER)
    return ds


def test_extract_roi_1(training_raster, training_roi):
    dat, label, xs, ys = next(roi.extract_roi(training_raster,
                                              training_roi,
                                              feature_prop='code_1',
                                              all_touched=False))
    geom = shapely.geometry.shape(training_roi[0]['geometry'])
    x_size, y_size = training_raster.res
    xy = np.stack((xs + int(x_size / 2), ys - int(y_size / 2)))

    contains = []
    for _xy in xy.T:
        p = shapely.geometry.Point(_xy)
        contains.append(geom.contains(p))
    assert sum(contains) / float(len(contains)) == 1.0


def test_extract_roi_2(training_raster, training_roi):
    # Test with ds.nodata included
    meta = training_raster.meta
    meta['nodata'] = 255
    meta['driver'] = 'MEM'
    with rasterio.open('tmp', 'w', **meta) as ds:
        dat, label, xs, ys = next(roi.extract_roi(ds,
                                                  training_roi,
                                                  feature_prop='code_1',
                                                  all_touched=False))
        geom = shapely.geometry.shape(training_roi[0]['geometry'])
        x_size, y_size = training_raster.res
        xy = np.stack((xs + int(x_size / 2), ys - int(y_size / 2)))

        contains = []
        for _xy in xy.T:
            p = shapely.geometry.Point(_xy)
            contains.append(geom.contains(p))
            assert sum(contains) / float(len(contains)) == 1.0
        assert dat.sum() == 0
