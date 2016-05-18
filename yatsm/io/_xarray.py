import logging

import numpy as np
import rasterio
import six
import xarray as xr

from . import _gdal

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


def read_and_preprocess(config, row0, col0, nrow=1, ncol=':'):
    """

    Note:
        To get a time series of a single pixel out of this:

    .. code:: python

        arr.isel(x=0, y=0).dropna('time')
    """
    datasets = {}
    for name, cfg in six.iteritems(config['data']['datasets']):
        logger.debug('Reading "{}" dataset'.format(name))
        arr = config_to_dataarray(cfg, row0, col0, nrow=nrow, ncol=ncol)

        if cfg['mask_band'] and cfg['mask_values']:
            logger.debug('Applying mask band to "{}"'.format(name))
            arr = apply_band_mask(arr, cfg['mask_band'], cfg['mask_values'])

        # Min/Max values -- done here for now
        if cfg['min_values'] and cfg['max_values']:
            logger.debug('Applying range mask to "{}"'.format(name))
            arr = apply_range_mask(arr, cfg['min_values'], cfg['max_values'])

        datasets[name] = arr

    return merge_datasets(datasets)


def config_to_dataarray(config, row0, col0, nrow=1, ncol=':'):
    """ Turns a data configuration section into xarray.DataArray
    """
    df = _gdal.parse_dataset_file(config['input_file'], config['date_format'])

    # TODO: get these from "driver" (GDAL / BIP / eventually, ARD connection)
    md = dict()
    with rasterio.drivers():
        with rasterio.open(df['filename'][0]) as src:
            md['affine'] = src.affine
            md['crs'] = src.crs
            md['nrow'] = src.height
            md['ncol'] = src.width
            md['vars'] = config['band_names']
            md['dtypes'] = src.dtypes
            assert len(config['band_names']) == src.count

    nrow = md['nrow'] if nrow == ':' else nrow
    ncol = md['ncol'] if ncol == ':' else ncol
    window = ((row0, row0 + nrow), (col0, col0 + ncol))
    nobs = len(df['date'])
    nband = len(config['band_names'])
    coord_y = md['affine'][2] + md['affine'][0] * np.arange(*window[0])
    coord_x = md['affine'][5] + md['affine'][4] * np.arange(*window[1])

    values = np.empty((nobs, nband, nrow, ncol), dtype=md['dtypes'][0])

    # Read time series
    # TODO: remove from here
    with rasterio.drivers():
        for i, row in df.iterrows():
            with rasterio.open(row['filename']) as src:
                src.read(window=window, out=values[i, ...], boundless=True)

    # Create DataArray
    da = xr.DataArray(
        values,
        dims=['time', 'band', 'y', 'x'],
        coords=[df['date'], config['band_names'], coord_y, coord_x]
    )
    return da


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
