""" Command line interface for running YATSM algorithms on individual pixels
"""
from datetime import datetime as dt
import logging
import os


import click
import palettable
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from yatsm.cli.cli import cli, config_file_arg
from yatsm.config_parser import parse_config_file
from yatsm.reader import find_stack_images, read_pixel_timeseries
from yatsm.regression.transforms import harm
from yatsm.yatsm import YATSM

plot_styles = []
if hasattr(mpl, 'style'):
    plot_styles = mpl.style.available
if hasattr(plt, 'xkcd'):
    plot_styles.append('xkcd')

logger = logging.getLogger('yatsm')


@cli.command(short_help='Run YATSM algorithm on individual pixels')
@config_file_arg
@click.argument('px', metavar='<px>', nargs=1, type=click.INT)
@click.argument('py', metavar='<py>', nargs=1, type=click.INT)
@click.option('--band', )
@click.pass_context
def pixel(ctx, config, px, py):
    # Parse config
    dataset_config, yatsm_config = parse_config_file(config)

