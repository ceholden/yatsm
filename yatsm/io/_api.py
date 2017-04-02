""" Functions, classes, etc. useful to CLI or other users
"""
from collections import OrderedDict
import datetime as dt
import logging

from ._xarray import (apply_band_mask, apply_range_mask, merge_data)
from .backends import READERS

logger = logging.getLogger(__name__)


def get_readers(config):
    """ Return a dict containing time series drivers described in config

    Args:
        config (dict): ``dataset`` entry in a YATSM configuration file with
            sections for each of the ``readers``

    Returns:
        dict: Time series drivers
    """
    drivers = OrderedDict()

    for name, cfg in config.items():
        logger.debug('Finding reader "{0}": {1}'.format(name, cfg))
        reader_name = cfg['reader']['name']
        reader_cfg = cfg['reader'].get('config', {})
        drivers[name] = get_reader(reader_name, **reader_cfg)

    return drivers


def get_reader(name, **kwds):
    """ Initialize a time series reader

    Function signature is flexible to allow for direct parameterization of
    a reader:

    .. code:: python

        # Using a data frame
        reader = get_reader('GDAL', df=df)
        # Using an input file
        reader = get_reader('GDAL', input_file='my_images.csv')

    or using a configuration dict:

    .. code:: python

        # Passed as kwargs
        cfg = {'input_file': 'my_images.csv'}
        reader = get_reader('GDAL', **cfg)

    Args:
        name (str): Optionally, provide the name of the backend
        kwds (dict): Keyword arguments used to create the driver, passed
            either to ``__init__`` or ``from_config``

    Raises:
        KeyError: when asking for a reader that doesn't exist
    """
    reader_cls = READERS.get(name, None)
    if not reader_cls:
        raise KeyError('Unknown time series reader: "{0}"'.format(name))

    if hasattr(reader_cls, 'from_config'):
        return reader_cls.from_config(**kwds)
    else:
        return reader_cls(**kwds)


def read_and_preprocess(config, readers, window, out=None):
    """ Read and preprocess a window of data from multiple readers

    Note:
        To get a time series of a single pixel out of this:

    .. code:: python

        arr.isel(x=0, y=0).dropna('time')

    Args:
        config (dict): ``dataset`` entry in a YATSM configuration file with
            sections for each of the ``readers``
        readers (list): A list of reader backends (e.g., ``GDAL``)
        window (tuple): A pair of (tuple) of pairs of ints specifying the
            start and stop indices of teh window rows and columns
        out (np.ndarray): A NumPy array of pre-allocated memory to read the
            time series into. Its shape should be:

            (time series length, # bands, # rows, # columns)

    Returns:
        xarray.Dataset: A ``Dataset`` contaiing all data and metadata from all
        drivers for the requested ``window``
    """
    datasets = {}
    for name, cfg in config.items():
        reader = readers[name]
        arr = reader.read_dataarray(window=window,
                                    bands=reader.band_names,
                                    out=out)

        if cfg['mask_band'] and cfg['mask_values']:
            logger.debug('Applying mask band to "{}"'.format(name))
            arr = apply_band_mask(arr, cfg['mask_band'], cfg['mask_values'])

        # Min/Max values -- done here for now
        if cfg['min_values'] and cfg['max_values']:
            logger.debug('Applying range mask to "{}"'.format(name))
            arr = apply_range_mask(arr, cfg['min_values'], cfg['max_values'])

        # Add in metadata
        md = reader.get_metadata()
        ds = arr.to_dataset(dim='band')
        for varname in ds.data_vars:  # attrs gone, so add them back in
            ds[varname].attrs = arr.attrs.copy()
        ds.update(md)

        datasets[name] = ds

    ds = merge_data(datasets)
    ds['doy'] = ('time', ds['time.dayofyear'])
    ds['ordinal'] = ('time', ds['time'].to_pandas().map(dt.datetime.toordinal))

    return ds
