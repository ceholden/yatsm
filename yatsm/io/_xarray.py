import logging

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def _where_with_attrs(arr, condition):
    _arr = arr.where(condition)
    _arr.attrs = arr.attrs
    return _arr


def apply_band_mask(arr, mask_band, mask_values):
    """ Mask all `bands` in `arr` based on some mask values in a band

    Args:
        arr (xarray.DataArray): Data array to mask
        mask_band (str): Name of `band` in `arr` to use for masking
        mask_values (sequence): Sequence of values to mask

    Returns:
        xarray.DataArray: Masked version of `arr`
    """
    _shape = (arr.time.size, arr.y.size, arr.x.size)
    mask = np.in1d(arr.sel(band=mask_band), mask_values,
                   invert=True).reshape(_shape)
    mask = xr.DataArray(mask, dims=['time', 'y', 'x'],
                        coords=[arr.time, arr.y, arr.x])

    return _where_with_attrs(arr, mask)


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

    return _where_with_attrs(arr, ((arr >= mins) & (arr <= maxs)))


def merge_datasets(dataarrays, merge_attrs=True):
    """ Combine multiple data arrays into one dataset

    Args:
        dataarrays (dict[name, xr.DataArray]): xr.DataArray objects to merge
        merge_attrs (bool): Attempt to merge DataArray attributes. In order for
            these attributes to be able to merge, they must be pd.Series
            and have compatible indexes.

    Returns:
        xr.Dataset: Merged xr.DataArray objects in one xr.Dataset
    """
    datasets = [arr.to_dataset(dim='band') for arr in dataarrays.values()]

    ds = datasets[0]
    ds_attrs = ds.attrs
    for _ds in datasets[1:]:
        ds = ds.merge(_ds, inplace=True)
        for attr in _ds.attrs.keys():
            if attr in ds_attrs:
                to_join = [ds.attrs[attr], _ds.attrs[attr]]
                if all([isinstance(a, pd.Series) for a in to_join]):
                    ds.attrs[attr] = pd.concat(to_join).sort_index()
            else:
                ds.attrs[attr] = (pd.Series(index=ds.time)
                                  .fillna(_ds.attrs[attr]))

    return ds
