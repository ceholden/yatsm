""" Command line interface for running YATSM pipelines in batch
"""
from collections import defaultdict, OrderedDict
import logging
from itertools import product
import os
import time

import click
import numpy as np
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
    from yatsm.pipeline import Pipe, Pipeline
    from yatsm.results import HDF5ResultsStore, result_filename

    config = validate_and_parse_configfile(configfile)

    readers = OrderedDict((
        (name, io_api.get_reader(**cfg['reader']))
        for name, cfg in config['data']['datasets'].items()
    ))

    # TODO: Better define how authoritative reader when using multiple datasets
    #       and choosing block shape (in config?)
    # TODO: Allow user to specify block shape in config (?)
    preference = next(iter(readers))
    from IPython.core.debugger import Pdb; Pdb().set_trace()
    block_windows = readers[preference].block_windows

    import dask

    def sel_pix(pipe, y, x):
        return Pipe(data=pipe['data'].sel(y=y, x=x),
                    record=pipe.get('record', None))

    overwrite = config['pipeline'].get('overwrite', True)
    tasks = config['pipeline']['tasks']

    # TODO: iterate over block_windows assigned to ``job_id``
    for idx, window in block_windows:
        data = io_api.read_and_preprocess(config['data']['datasets'],
                                          readers,
                                          ((0, 10), (0, 10)),
                                          out=None)

        filename = result_filename(
            window,
            root=config['results']['output'],
            pattern=config['results']['output_prefix'],
        )
        with HDF5ResultsStore(filename) as result_store:

            # TODO: read this from pre-existing results
            pipe = Pipe(data=data)
            pipeline = Pipeline.from_config(tasks, pipe, overwrite=overwrite)
            pipe = pipeline.run_eager(pipe)

            record_results = defaultdict(list)
            for y, x in product(data.y.values, data.x.values):
                logger.debug('Processing pixel y/x: {}/{}'.format(y, x))
                pix_pipe = sel_pix(pipe, y, x)

                result = pipeline.run(pix_pipe)

                # TODO: figure out what to do with 'data' results
                for k, v in result['record'].items():
                    record_results[k].append(v)

            for name, result in record_results.items():
                record_results[name] = np.concatenate(result)
            result_store.write_result(pipeline, record_results,
                                      overwrite=overwrite)

