""" Command line interface for classifying YATSM algorithm output
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np

from yatsm.cli import options

logger = logging.getLogger('yatsm')


@click.command(short_help='Classify entire images using trained algorithm')
@click.pass_context
def classify(ctx):
    raise NotImplementedError('TODO')
