from functools import partial
import logging
import os

import numpy as np
import pandas as pd
import rasterio
import six
import xarray as xr

logger = logging.getLogger(__name__)


def _read_dataset_input_file(input_file, date_format):
    """ Return parsed dataset CSV file as pd.DataFrame

    Args:
        input_file (str): CSV filename
        date_format (str): Format of date in input file

    Returns:
        pd.DataFrame: Dataset information
    """
    dt_parser = lambda x: pd.datetime.strptime(x, date_format)
    df = pd.read_csv(input_file,
                     parse_dates=['date'], date_parser=dt_parser)
    if not os.path.isabs(df['filename'][0]):
        _root = os.path.abspath(os.path.dirname(input_file))
        _root_join = partial(os.path.join, _root)
        df['filename'] = map(_root_join, df['filename'])

    return df


def config_to_dataarray(config, row0, col0, nrow=1, ncol=':'):
    """ Turns a data configuration section into xarray.DataArray
    """
    df = _read_dataset_input_file(config['input_file'], config['date_format'])

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


def read_and_preprocess(config, row0, col0):
    datasets = {}
    for name, cfg in six.iteritems(config['data']['datasets']):
        logger.debug('Reading "{}" dataset'.format(name))
        arr = config_to_dataarray(cfg, row0, col0, nrow=1, ncol=1)

        # Mask -- done here for now
        if cfg['mask_band']:
            mask = np.in1d(arr.sel(band=cfg['mask_band']), cfg['mask_values'],
                           invert=True)
            arr = arr.isel(time=mask)

        # Min/Max values -- done here for now
        if cfg['max_values'] and cfg['min_values']:
            # If we turn these into DataArrays, magic happens
            mins = xr.DataArray(cfg['min_values'], dims=['band'],
                                coords=[cfg['band_names']])
            maxs = xr.DataArray(cfg['max_values'], dims=['band'],
                                coords=[cfg['band_names']])

            mask = np.ones((arr.time.size,
                            arr.band.size,
                            arr.y.size,
                            arr.x.size)).astype(np.bool)

            arr = arr.where((arr >= mins) & (arr <= maxs))
        datasets[name] = arr

    return merge_datasets(datasets)


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
