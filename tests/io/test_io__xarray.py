""" Tests for yatsm.io._xarray
"""
import numpy as np
import pytest

import yatsm.io._xarray as _xr


MASK_BAND = 'fmask'

fixture_numeric_range = pytest.mark.parametrize(
    ('mins', 'maxs'), (
        (0, 10000),
        (-2000, 10000),
        (-2000, 16000),
        (0, 10000),
        (-2000, 16000),
    )
)


fixture_dict_range = pytest.mark.parametrize(
    ('mins', 'maxs'), (
        ({'blue': 0}, {'blue': 10000}),
        ({'red': -2000}, {'red': 16000}),
        ({'blue': -2000, 'green': 0, 'swir1': 0},
         {'blue': 16000, 'green': 16000, 'swir1': 16000}),
    )
)


def test_apply_mask_band_nodrop(HRF1_da):
    arr = _xr.apply_band_mask(
        HRF1_da,
        MASK_BAND,
        [2, 3, 4, 255],
        drop=False
    )
    assert arr.sel(band=MASK_BAND).max() <= 1
    assert all(arr.band == HRF1_da.band)
    # Same shape since no drop
    assert arr.shape == HRF1_da.shape


def test_apply_mask_band_drop(HRF1_da):
    arr = _xr.apply_band_mask(
        HRF1_da,
        MASK_BAND,
        [2, 3, 4, 255],
        drop=True
    )
    assert arr.sel(band=MASK_BAND).max() <= 1
    assert all(arr.band == HRF1_da.band)
    # Time dimension at least as short
    assert arr.time.size <= HRF1_da.time.size


@fixture_numeric_range
def test_apply_range_mask_numeric_nodrop(HRF1_da, mins, maxs):
    arr = _xr.apply_range_mask(
        HRF1_da,
        mins,
        maxs,
        drop=False
    )

    assert arr.max() <= maxs
    assert arr.min() >= mins
    assert all(arr.band == HRF1_da.band)
    assert arr.time.size == HRF1_da.time.size


@fixture_numeric_range
def test_apply_range_mask_numeric_drop(HRF1_da, mins, maxs):
    arr = _xr.apply_range_mask(
        HRF1_da,
        mins,
        maxs,
        drop=True
    )

    assert arr.max() <= maxs
    assert arr.min() >= mins
    assert all(arr.band == HRF1_da.band)
    assert arr.time.size <= HRF1_da.time.size


@fixture_dict_range
def test_apply_range_mask_dict_drop(HRF1_da, mins, maxs):
    # no drop
    arr = _xr.apply_range_mask(
        HRF1_da,
        mins,
        maxs,
        drop=False
    )

    for band in mins:
        assert arr.loc[band].min() >= mins[band]
    for band in maxs:
        assert arr.loc[band].max() <= maxs[band]
    assert all(arr.band == HRF1_da.band)
    assert arr.time.size == HRF1_da.time.size


@fixture_dict_range
def test_apply_range_mask_dict_nodrop(HRF1_da, mins, maxs):
    # drop with 'all' required to drop an observation
    # this shouldn't change
    arr = _xr.apply_range_mask(
        HRF1_da,
        mins,
        maxs,
        drop=True
    )

    for band in mins:
        assert arr.loc[band].min() >= mins[band]
    for band in maxs:
        assert arr.loc[band].max() <= maxs[band]
    assert all(arr.band == HRF1_da.band)
    assert arr.time.size <= HRF1_da.time.size
