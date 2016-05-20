""" Command line interface for running YATSM on image lines """
from collections import OrderedDict
import logging
import os
import time

import click
import numpy as np
import six

from . import options

from ..errors import TSLengthException
from ..version import __version__

logger = logging.getLogger('yatsm')


@click.command(short_help='Run YATSM on an entire image line by line')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.option('--check_cache', is_flag=True,
              help='Check that cache file contains matching data')
@click.option('--resume', is_flag=True,
              help='Do not overwrite preexisting results')
@click.option('--do-not-run', is_flag=True,
              help='Do not run YATSM (useful for just caching data)')
@click.pass_context
def line(ctx, configfile, job_number, total_jobs,
         resume, check_cache, do_not_run):
    # Imports inside CLI for speed
    from ..config import validate_and_parse_configfile
    from ..io import _api as io_api

    # Parse config
    config = validate_and_parse_configfile(configfile)
    # Find readers for datasets
    readers = OrderedDict((
        (name, io_api.get_reader(**cfg['reader'])) for name, cfg
        in six.iteritems(config['data']['datasets'])
    ))

    blocks = (250, 250)
    windows = [
        ((0, 250), (0, 250)),
    ]

    for window in windows:
        data = io_api.read_and_preprocess(config['data']['datasets'],
                                          readers, window, out=None)
        from IPython.core.debugger import Pdb; Pdb().set_trace()
