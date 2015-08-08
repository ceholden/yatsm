""" Command line interface for working with cached data for YATSM algorithms
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np

from yatsm.cli import options

logger = logging.getLogger('yatsm')


@click.command(short_help='Create or update cached timeseries data for YATSM')
@click.pass_context
def cache(ctx):
    raise NotImplementedError('TODO')
