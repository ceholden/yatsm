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
from ..pipeline import config_to_tasks
from ..version import __version__

logger = logging.getLogger('yatsm')


@click.command(short_help='Run YATSM on an entire image line by line')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.pass_context
def line(ctx, configfile, job_number, total_jobs):
    # Imports inside CLI for speed
    from yatsm.config import validate_and_parse_configfile
    from yatsm.io import _api as io_api

    config = validate_and_parse_configfile(configfile)
    readers = OrderedDict((
        (name, io_api.get_reader(**cfg['reader'])) for name, cfg
        in six.iteritems(config['data']['datasets'])
    ))

    # TODO: Better define how authoritative reader when using multiple datasets
    #       and choosing block shape (in config?)
    # TODO: Allow user to specify block shape in config (?)
    preference = next(iter(readers))
    block_windows = readers[preference].block_windows

    for idx, window in block_windows:
        data = io_api.read_and_preprocess(config['data']['datasets'],
                                          readers, window, out=None)
        pipe = {
            'data': data,
            'record': {}  # TODO: read this from pre-existing results
        }
        tasks = config_to_tasks(config['pipeline']['tasks'], pipe)
        from IPython.core.debugger import Pdb; Pdb().set_trace()
