""" Functions, classes, etc. useful to CLI or other users
"""
from collections import OrderedDict
import logging

import six

from ._xarray import (apply_band_mask, apply_range_mask, merge_datasets)
from .backends import READERS

logger = logging.getLogger(__name__)


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
        pass
    elif name is None and kwargs:
        name, kwargs = kwargs.popitem()
    else:
        raise ValueError('"name" must either be a `str` or None, with '
                         'reader options specified by kwargs')


    reader_cls = READERS.get(name, None)
    if not reader_cls:
        raise KeyError('Unknown time series reader "{}"'.format(name))

    return reader_cls(**kwargs)



# TODO: this function should take in one or more readers and a window
def read_and_preprocess(config, row0, col0, nrow=1, ncol=':', out=None):
    """

    Note:
        To get a time series of a single pixel out of this:

    .. code:: python

        arr.isel(x=0, y=0).dropna('time')
    """
    # TODO: calculate windows elsewhere
    windows = [((row0, row0 + nrow), (col0, col0 + ncol))]
    # TODO: iterate over windows?
    window = windows[0]

    datasets = {}
    for name, cfg in six.iteritems(config['data']['datasets']):
        if cfg['reader'] in READERS:
            reader = READERS[cfg['reader']](**cfg)
            arr = reader.read_dataarray(window=window, out=out)
        else:
            raise KeyError('"{}" reader backend is not supported at this time'
                           .format(cfg['reader']))

        if cfg['mask_band'] and cfg['mask_values']:
            logger.debug('Applying mask band to "{}"'.format(name))
            arr = apply_band_mask(arr, cfg['mask_band'], cfg['mask_values'])

        # Min/Max values -- donberline here for now
        if cfg['min_values'] and cfg['max_values']:
            logger.debug('Applying range mask to "{}"'.format(name))
            arr = apply_range_mask(arr, cfg['min_values'], cfg['max_values'])

        datasets[name] = arr

    return merge_datasets(datasets)
