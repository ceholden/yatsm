""" Read/write tasks
"""
import logging

import six
import xarray as xr

from yatsm.io import merge_data
from yatsm.pipeline import eager_task, task_version, outputs

logger = logging.getLogger(__name__)


@task_version('xarray_open:1.0.0')
@eager_task
@outputs(data=[])
def xarray_open(pipe, require, output, config=None):
    """ Open a dataset with xarray

    Args:
        filename (str): Filename to read
    """
    filename = config.pop('filename')

    if isinstance(filename, six.string_types) and '*' not in filename:
        logger.debug('Opening `xr.open_dataset`: {0}'.format(filename))
        ds = xr.open_dataset(filename, **config)
    if isinstance(filename, six.string_types) and '*' in filename:
        logger.debug('Opening files `xr.open_mfdataset`: {0}'
                     .format(filename))
        ds = xr.open_mfdataset(filename, **config)
    elif isinstance(filename, (list, tuple)):
        logger.debug('Opening multiple files `xr.open_mfdataset`: {0}'
                     .format(', '.join(filename)))
        ds = xr.open_mfdataset(filename, **config)

    if pipe.data:  # need to merge
        logger.debug('Need to merge data from "{0}" to add it into '
                     'existing pipe data ("{1}")'
                     .format(ds, pipe.data))
        merged = merge_data(pipe.data, ds)
        pipe.data = merged
    else:
        pipe.data = ds
    return pipe
