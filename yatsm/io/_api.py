""" Functions, classes, etc. useful to CLI or other users
"""
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



def read_and_preprocess(config, readers, window, out=None):
    """

    Note:
        To get a time series of a single pixel out of this:

    .. code:: python

        arr.isel(x=0, y=0).dropna('time')
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

        # Min/Max values -- donberline here for now
        if cfg['min_values'] and cfg['max_values']:
            logger.debug('Applying range mask to "{}"'.format(name))
            arr = apply_range_mask(arr, cfg['min_values'], cfg['max_values'])

        # Add in metadata
        md = reader.get_metadata().to_dataset(dim='band')
        ds = arr.to_dataset(dim='band')
        ds = ds.assign_coords(**ds.coords.merge(md))
        # ds.update(md)
        # ds.coords.set_coords(ds.coords.keys() + md.data_vars.keys())

        datasets[name] = ds

    return merge_datasets(datasets)
