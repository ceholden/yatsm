""" Command line interface for running YATSM pipelines in batch
"""
from __future__ import division

from collections import defaultdict
import logging
from itertools import product
import time

import click

from . import options

logger = logging.getLogger('yatsm')


@click.command(short_help='Run a YATSM pipeline on a dataset in batch mode')
@options.arg_config
@options.arg_job_number
@options.arg_total_jobs
@options.opt_executor
@options.opt_force_overwrite
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
    from yatsm.utils import distribute_jobs

    # TODO: remove when not debugging
    import dask
    dask.set_options(get=dask.async.get_sync)


    # TODO: Better define how authoritative reader when using multiple datasets
    #       and choosing block shape (in config?)
    # TODO: Allow user to specify block shape in config (?)
    block_windows = config.primary_reader.block_windows
    job_idx = distribute_jobs(job_number, total_jobs, len(block_windows))

    logger.debug('Working on {} of {} block windows'
                 .format(len(job_idx), len(block_windows)))

    block_windows = [block_windows[i] for i in job_idx]

    force_overwrite = (force_overwrite or
                       config['pipeline'].get('overwrite', False))

    # TODO: iterate over block_windows assigned to ``job_id``
    futures = {}
    for idx, window in block_windows:
        future = executor.submit(batch_block,
                                 config=config,
                                 readers=config.readers,
                                 window=window,
                                 overwrite=force_overwrite)
        futures[future] = window

    n_good, n_skip, n_fail = 0, 0, 0
    for future in executor.as_completed(futures):
        window = futures[future]
        try:
            result = future.result()
            if isinstance(result, str):
                logger.info("Wrote to: %s" % result)
                n_good += 1
            else:
                n_skip += 1
            time.sleep(1)
        except KeyboardInterrupt:
            logger.critical('Interrupting and shutting down')
            executor.shutdown()
            raise click.Abort()
        except Exception:
            logger.exception("Exception for window: {}".format(window))
            n_fail += 1
            raise  # TODO: remove and log?

    logger.info('Complete: %s' % n_good)
    logger.info('Skipped: %s' % n_skip)
    logger.info('Failed: %s' % n_fail)


def batch_block(config, readers, window, overwrite=False):
    import logging

    import numpy as np

    from yatsm import io
    from yatsm.results import HDF5ResultsStore
    from yatsm.pipeline import Pipe

    logger = logging.getLogger('yatsm')

    def sel_pix(pipe, y, x):
        return Pipe(data=pipe['data'].sel(y=y, x=x),
                    record=pipe.get('record', None))

    logger.info('Working on window: {}'.format(window))
    data = io.read_and_preprocess(config['data']['datasets'],
                                  readers,
                                  window,
                                  out=None)

    store_kwds = {
        'window': window,
        'reader': config.primary_reader,
        'root': config['results']['output'],
        'pattern': config['results']['output_prefix'],
    }

    # TODO: guess for number of records to store
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    with HDF5ResultsStore.from_window(**store_kwds) as store:
        # TODO: read this from pre-existing results
        pipe = Pipe(data=data)
        pipeline = config.get_pipeline(pipe, overwrite=overwrite)
        from IPython.core.debugger import Pdb; Pdb().set_trace()

        # TODO: finish checking for resume
        if store.completed(pipeline) and not overwrite:
            logger.info('Already completed: {}'.format(store.filename))
            return

        pipe = pipeline.run_eager(pipe)

        record_results = defaultdict(list)
        n_ = data.y.shape[0] * data.x.shape[0]
        for i, (y, x) in enumerate(product(data.y.values, data.x.values)):
            logger.debug('Processing pixel {pct:>4.2f}%: y/x {y}/{x}'
                         .format(pct=i / n_ * 100, y=y, x=x))
            pix_pipe = sel_pix(pipe, y, x)

            result = pipeline.run(pix_pipe, check_eager=False)

            # TODO: figure out what to do with 'data' results
            for k, v in result['record'].items():
                record_results[k].append(v)

        for name, result in record_results.items():
            record_results[name] = np.concatenate(result)

        if record_results:
            store.write_result(pipeline, record_results,
                               overwrite=overwrite)
        # TODO: write out cached data
        return store.filename

