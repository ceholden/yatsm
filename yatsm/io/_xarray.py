""" Helper functions for :ref:`xarray.Dataset` and `xarray.DataArray` objects
"""
from collections import OrderedDict
import logging
import warnings

import numpy as np
import xarray as xr

from yatsm.errors import TODO
from yatsm.gis import CRS, share_crs

logger = logging.getLogger(__name__)


def apply_band_mask(arr, mask_band, mask_values, drop=False):
    """ Mask all `band` in `arr` based on some mask values in a band

    Args:
        arr (xarray.DataArray): Data array to mask
        mask_band (str): Name of `band` in `arr` to use for masking
        mask_values (sequence): Sequence of values to mask
        drop (bool): Drop observations masked by :meth:`xr.Dataset.where`

    Returns:
        xarray.DataArray: Masked version of `arr`
    """
    _dims = ('time', 'y', 'x', )  # TODO: define these coords somewhere?
    dims = tuple((dim for dim in _dims if dim in arr.dims))
    shape = tuple((arr[dim].size for dim in dims))
    coords = tuple((arr[dim] for dim in dims))

    mask = np.in1d(arr.sel(band=mask_band), mask_values,
                   invert=True).reshape(shape)
    mask = xr.DataArray(mask, dims=dims, coords=coords)

    return arr.where(mask, drop=drop)


def apply_range_mask(arr, min_values, max_values, drop=False, how='all'):
    """ Mask a DataArray based on a range of acceptable values

    Minimum and maximum values may be passed in one of three ways:

        1. Pass a single number and the function will use this number as the
           minimum/maximum value for all bands
        2. Pass a dictionary, and the range mask will be applied on these bands
        3. Pass a sequence of numbers that is the same length as the number of
           bands in ``arr``

    Args:
        arr (xarray.DataArray): Data array to mask
        min_values (float/int, sequence, or dict): Minimum values
        max_values (float/int, sequence, or dict): Maximum values
        drop (bool): Drop observations masked by :meth:`xr.Dataset.where`
        how (str): Drop or mask observations across bands depending if
            `any` or `all` of the observations are out of range. Using
            `any` will drop as many or more observations than `all`.

    Returns:
        xarray.DataArray: Masked version of `arr`
    """
    assert 'time' in arr.dims
    assert how in ('any', 'all'), "`how` must be `all` or `any`"

    def _parse(arr, value):
        if isinstance(value, (int, float)):
            return OrderedDict(((band.item(), value) for band in arr.band))
        elif isinstance(value, dict):
            return value
        elif isinstance(value, (tuple, list)):
            if len(value) != len(arr.band):
                raise ValueError('Must provide a value for each band when '
                                 'using a `list` or `tuple`. Got {n} values '
                                 'for an array with {b} bands'
                                 .format(n=len(value), b=len(arr.band)))
            return dict(zip(arr.band.values, value))
        else:
            raise TypeError('Must specify `min_values` or `max_values` as '
                            'either a number (int, float), a sequence of '
                            'numbers, or as a `dict` mapping band names '
                            'to numbers')

    dt_info = (np.iinfo(arr.dtype) if arr.dtype.kind == 'i' else
               np.finfo(arr.dtype))

    _mins = _parse(arr, min_values)
    _maxs = _parse(arr, max_values)

    bands = set(_mins).union(set(_maxs))

    mins = OrderedDict(((b, _mins.get(b, dt_info.min)) for b in bands))
    maxs = OrderedDict(((b, _maxs.get(b, dt_info.max)) for b in bands))

    # If we turn these into DataArrays, magic (axis alignment) happens
    mins = xr.DataArray(np.asarray(list(mins.values()), dtype=arr.dtype),
                        dims=['band'], coords={'band': list(mins)})
    maxs = xr.DataArray(np.asarray(list(maxs.values()), dtype=arr.dtype),
                        dims=['band'], coords={'band': list(maxs)})

    # Silence gt/lt/ge/le/eq with nan.
    # See: http://stackoverflow.com/q/41130138
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)

        # Remember we want the inverse because mask==1 is "good" obs
        mask = ((arr >= mins) & (arr <= maxs))
        mask = (mask.any(dim='band') if how == 'any'  # all bad to drop
                else mask.all(dim='band'))  # all good to keep

        return arr.where(mask, drop=drop)


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
