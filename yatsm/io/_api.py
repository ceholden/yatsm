""" Functions, classes, etc. useful to CLI or other users
"""
from collections import OrderedDict
import logging

import six

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
    return OrderedDict((
        (name, get_reader(**cfg['reader'])) for name, cfg in config.items()
    ))


def get_reader(name=None, **kwargs):
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
        reader = get_reader(**{'GDAL': {'input_file': 'my_images.csv'}})

    Args:
        name (str): Optionally, provide the name of the backend

    Raises:
        ValueError: if `name` and **kwargs aren't properly specified
        KeyError: when asking for a reader that doesn't exist
    """
    if isinstance(name, six.string_types) and isinstance(kwargs, dict):
        if kwargs.keys() == [name]:
            kwargs = kwargs[name]
    elif name is None and kwargs:
        name, kwargs = kwargs.popitem()
    else:
        raise ValueError('"name" must either be a `str` or None, with '
                         'reader options specified by kwargs')

    reader_cls = READERS.get(name, None)
    if not reader_cls:
        raise KeyError('Unknown time series reader "{}"'.format(name))

    return reader_cls(**kwargs)


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
    for name, cfg in six.iteritems(config):
        reader = readers[name]
        arr = reader.read_dataarray(window=window,
                                    band_names=cfg['band_names'],
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
        ds.update(md)

        datasets[name] = ds

    return merge_data(datasets)
