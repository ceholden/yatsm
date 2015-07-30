""" Command line interface for creating changemaps of YATSM algorithm output
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np

from yatsm.cli.cli import cli

logger = logging.getLogger('yatsm')


@cli.command(short_help='Map change found by YATSM algorithm over time period')
@click.pass_context
def changemap(ctx):
    raise NotImplementedError('TODO')
