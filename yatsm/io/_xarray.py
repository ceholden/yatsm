""" Helper functions for :ref:`xarray.Dataset` and `xarray.DataArray` objects
"""
import logging
import warnings

import numpy as np
import xarray as xr

from yatsm.errors import TODO
from yatsm.gis import CRS, share_crs

logger = logging.getLogger(__name__)


def apply_band_mask(arr, mask_band, mask_values):
    """ Mask all `band` in `arr` based on some mask values in a band

    Args:
        arr (xarray.DataArray): Data array to mask
        mask_band (str): Name of `band` in `arr` to use for masking
        mask_values (sequence): Sequence of values to mask

    Returns:
        xarray.DataArray: Masked version of `arr`
    """
    _dims = {'time', 'y', 'x'}  # TODO: define these coords somewhere?
    _shape = (arr[dim].size for dim in _dims.intersection(arr.dims))
    mask = np.in1d(arr.sel(band=mask_band), mask_values,
                   invert=True).reshape(_shape)
    mask = xr.DataArray(mask, dims=['time', 'y', 'x'],
                        coords=[arr.time, arr.y, arr.x])

    return arr.where(mask)


def apply_range_mask(arr, min_values, max_values):
    """ Mask a DataArray based on a range of acceptable values

    Args:
        arr (xarray.DataArray): Data array to mask
        min_values (sequence): Minimum values per `band` in `arr`
        max_values (sequence): Maximum values per `band` in `arr`

    Returns:
        xarray.DataArray: Masked version of `arr`
    """
    # If we turn these into DataArrays, magic happens
    maxs = xr.DataArray(np.asarray(max_values, dtype=arr.dtype),
                        dims=['band'], coords=[arr.coords['band']])
    mins = xr.DataArray(np.asarray(min_values, dtype=arr.dtype),
                        dims=['band'], coords=[arr.coords['band']])

    # Silence gt/lt/ge/le/eq with nan. See: http://stackoverflow.com/q/41130138
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return arr.where(((arr >= mins) & (arr <= maxs)))


def merge_data(data, merge_attrs=True):
    """ Combine multiple Datasets or DataArrays into one Dataset

    Args:
        data (dict[name, xr.DataArray or xr.Dataset]): xr.DataArray or
            xr.Dataset objects to merge
        merge_attrs (bool): Attempt to merge DataArray attributes. In order for
            these attributes to be able to merge, they must be pd.Series
            and have compatible indexes.

    Returns:
        xr.Dataset: Merged xr.DataArray objects in one xr.Dataset
    """
    # TODO: (re)projections
    ds_crs = [CRS.from_string(data[ds].attrs['crs_wkt']) for ds in data]
    if not share_crs(*ds_crs):
        raise TODO('Cannot merge data with different CRS')

    datasets = [dat.to_dataset(dim='band') if isinstance(dat, xr.DataArray)
                else dat for dat in data.values()]

    # TODO: be helpful when this fails with good error
    ds = xr.merge(datasets, compat='no_conflicts')

    # Put attrs back on, first is authoritative
    ds.attrs = datasets[0].attrs.copy()
    for dataset in datasets[1:]:
        for attr in dataset.attrs:
            if attr not in ds.attrs:
                ds.attrs[attr] = dataset[attr]

    # TODO: probably going to need some long help message with try/except block
    #       since merging could be hard
    return ds
