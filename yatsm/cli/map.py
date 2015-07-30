""" Command line interface for creating maps of YATSM algorithm output
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np

from yatsm.cli.cli import cli

logger = logging.getLogger('yatsm')


@cli.command(short_help='Make map of YATSM output for a given date')
@click.pass_context
def map(ctx):
    """
    Examples:
    > yatsm_map.py --coef "intercept, slope" --band "3, 4, 5" --ndv -9999 coef
    ... 2000-01-01 coef_map.gtif

    > yatsm_map.py --date "%Y-%j" predict 2000-001 prediction.gtif

    > yatsm_map.py --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

    Notes:
        - Image predictions will not use categorical information in timeseries
          models.
    """
    raise NotImplementedError('TODO')
