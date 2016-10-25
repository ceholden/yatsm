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
    datasets = [dat.to_dataset(dim='band') if isinstance(dat, xr.DataArray)
                else dat for dat in data.values()]

    ds = xr.merge(datasets, compat='minimal')

    # Overlapping but not conflicting variables can't be merged for now
    # https://github.com/pydata/xarray/issues/835
    # In meantime, check for dropped variables
    dropped = set()
    for _ds in datasets:
        dropped.update(set(_ds.data_vars).difference(set(ds.data_vars)))

    # TODO: refactor this and repeat for coords and vars
    if dropped:
        # dropped_vars = {}
        for var in dropped:
            dims = [_ds[var].dims for _ds in datasets if var in _ds]
            if all([dims[0] == d for d in dims[1:]]):
                # Combine
                dfs = pd.concat([_ds[var].to_series()
                                 for _ds in datasets
                                 if var in _ds]).sort_index()
                # Recreate with same index as `ds`
                idx = ds.indexes[dfs.index.name]
                dfs = dfs[~dfs.index.duplicated()].reindex(idx)
                # Insert
                ds[var] = xr.DataArray(dfs)
            else:
                logger.debug("Cannot restore dropped coord {} because "
                             "dimensions are inconsistent across datasets")
        # ds = ds.assign_coords(**ds.coords.merge(dropped_vars))

    return ds
