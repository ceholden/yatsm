""" Command line interface for running YATSM pipelines in batch
"""
from collections import defaultdict, OrderedDict
import logging
from itertools import product
import os
import time

import click
import numpy as np
from rasterio.rio import options as rio_options
import six

from . import options

logger = logging.getLogger('yatsm')


@click.command(short_help='Run a YATSM pipeline on a dataset in batch mode')
@options.arg_config
@options.arg_job_number
@options.arg_total_jobs
@options.opt_executor
@rio_options.force_overwrite_opt
@click.pass_context
def batch(ctx, config, job_number, total_jobs, executor, force_overwrite):
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
    from yatsm.io import _api as io_api
    from yatsm.utils import distribute_jobs
    from yatsm.pipeline import Pipe, Pipeline

    # TODO: remove when not debugging
    import dask
    dask.set_options(get=dask.async.get_sync)

    readers = OrderedDict((
        (name, io_api.get_reader(**cfg['reader']))
        for name, cfg in config['data']['datasets'].items()
    ))

    from IPython.core.debugger import Pdb; Pdb().set_trace()

    # TODO: Better define how authoritative reader when using multiple datasets
    #       and choosing block shape (in config?)
    # TODO: Allow user to specify block shape in config (?)
    preference = next(iter(readers))
    block_windows = readers[preference].block_windows
    job_idx = distribute_jobs(job_number, total_jobs, len(block_windows))

    logger.debug('Working on {} of {} block windows'
                 .format(len(job_idx), len(block_windows)))

    block_windows = [block_windows[i] for i in job_idx]

    force_overwrite = (force_overwrite or
                       config['pipeline'].get('overwrite', False))
    tasks = config['pipeline']['tasks']

    # TODO: iterate over block_windows assigned to ``job_id``
    futures = {}
    from IPython.core.debugger import Pdb; Pdb().set_trace()
    for idx, window in block_windows:
        future = executor.submit(batch_block,
                                 config=config,
                                 readers=readers,
                                 window=window,
                                 overwrite=force_overwrite)
        futures[future] = window

    for result in executor.as_completed(futures):
        window = futures[result]
        print(result.result())
        from IPython.core.debugger import Pdb; Pdb().set_trace()
#        try:
#            pass  # TODO
#        except:
#            pass  # TODO
#


def batch_block(config, readers, window, overwrite=False):
    from yatsm.io import _api as io_api
    from yatsm.results import HDF5ResultsStore, result_filename
    from yatsm.pipeline import Pipe, Pipeline

    def sel_pix(pipe, y, x):
        return Pipe(data=pipe['data'].sel(y=y, x=x),
                    record=pipe.get('record', None))

    logger.debug('Working on window: {}'.format(window))
    data = io_api.read_and_preprocess(config['data']['datasets'],
                                      readers,
                                      window,
                                      out=None)

    filename = result_filename(
        window,
        root=config['results']['output'],
        pattern=config['results']['output_prefix'],
    )

    # TODO: guess for number of records to store
    with HDF5ResultsStore(filename) as result_store:
        # TODO: read this from pre-existing results
        pipe = Pipe(data=data)
        pipeline = Pipeline.from_config(config['pipeline']['tasks'], pipe,
                                        overwrite=overwrite)

        # TODO: finish checking for resume
        if all([table_name in result_store.keys() for table_name in
                pipeline.task_tables.values()]) and not overwrite:
            logger.info('Already completed: {}'.format(filename))
            return
            # continue


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

        if record_results:
            result_store.write_result(pipeline, record_results,
                                      overwrite=overwrite)

        # TODO: write out cached data
