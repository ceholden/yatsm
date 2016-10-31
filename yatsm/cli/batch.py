""" Command line interface for running YATSM pipelines in batch
"""
from collections import OrderedDict
import logging
from itertools import product
import os
import time

import click
import six
import toolz

from . import options
from ..errors import TSLengthException

logger = logging.getLogger('yatsm')


@click.command(short_help='Run a YATSM pipeline on a dataset in batch mode')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.pass_context
def batch(ctx, configfile, job_number, total_jobs):
    """ Run a YATSM pipeline on a dataset in batch mode

    The dataset is split into a number of subsets based on the structure of the
    files in the dataset. The internal structure is determined by the block
    sizes, or internal tile sizes, retrieved by GDAL. In the absence of the
    dataset being tiled, GDAL will default to 256 pixels in the X dimension and
    a value in the Y dimension that ensures that the block fits in 8K or less.

    TODO: Users may override the size of the subsets using command line
          options.
    """
    # Imports inside CLI for speed
    from yatsm.config import validate_and_parse_configfile
    from yatsm.io import _api as io_api
    from yatsm.pipeline import Pipeline

    config = validate_and_parse_configfile(configfile)

    readers = OrderedDict((
        (name, io_api.get_reader(**cfg['reader']))
        for name, cfg in config['data']['datasets'].items()
    ))

    # TODO: Better define how authoritative reader when using multiple datasets
    #       and choosing block shape (in config?)
    # TODO: Allow user to specify block shape in config (?)
    preference = next(iter(readers))
    block_windows = readers[preference].block_windows

    import dask

    def sel_pix(pipe, y, x):
        return {
            'data': pipe['data'].sel(y=y, x=x),
            'record': pipe['record']  # TODO: select pixel
        }

    overwrite = config['pipeline'].get('overwrite', True)
    tasks = config['pipeline']['tasks']

    # TODO: iterate over block_windows assigned to ``job_id``
    for idx, window in block_windows:
        data = io_api.read_and_preprocess(config['data']['datasets'],
                                          readers,
                                          ((0, 10), (0, 10)),
                                          out=None)
        pipe = {
            'data': data,
            'record': {}  # TODO: read this from pre-existing results
        }
        pipeline = Pipeline.from_config(tasks, pipe, overwrite=overwrite)
        pipe = pipeline.run_eager(pipe)

        for y, x in product(data.y.values, data.x.values):
            logger.debug('Processing pixel y/x: {}/{}'.format(y, x))
            pix_pipe = sel_pix(pipe, y, x)
            result = pipeline.run(pix_pipe)
            # TODO: save result...
