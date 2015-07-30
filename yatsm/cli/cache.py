""" Command line interface for working with cached data for YATSM algorithms
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np

from yatsm.cli.cli import cli

logger = logging.getLogger('yatsm')


@cli.command(short_help='Create or update cached timeseries data for YATSM')
@click.pass_context
def cache(ctx):
    raise NotImplementedError('TODO')
