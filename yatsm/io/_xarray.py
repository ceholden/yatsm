import logging

import numpy as np
import six
import xarray as xr

logger = logging.getLogger(__name__)


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
    maxs = xr.DataArray(max_values, dims=['band'], coords=[arr.coords['band']])
    mins = xr.DataArray(min_values, dims=['band'], coords=[arr.coords['band']])

    return arr.where((arr >= mins) & (arr <= maxs))



def merge_datasets(dataarrays):
    """
    Args:
        dataarrays (dict): name: xr.DataArray
    """
    datasets = [arr.to_dataset(dim='band') for arr in dataarrays.values()]

    ds = datasets[0]
    for _ds in datasets[1:]:
        ds = ds.merge(_ds, overwrite_vars=['time', 'y', 'x'])

    return ds
