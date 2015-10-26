""" Loads all commands for YATSM command line interface

Modeled after very nice `click` interface for `rasterio`:
https://github.com/mapbox/rasterio/blob/master/rasterio/rio/main.py

"""
import logging
import os
from pkg_resources import iter_entry_points

import click
import click_plugins

import yatsm
from . import options

NP_THREAD_VARS = ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPM_NUM_THREADS']

# YATSM CLI group
_context = dict(
    token_normalize_func=lambda x: x.lower(),
    help_option_names=['--help', '-h']
)


@click_plugins.with_plugins(ep for ep in
                            iter_entry_points('yatsm.yatsm_commands'))
@click.group(help='YATSM command line interface', context_settings=_context)
@click.version_option(yatsm.__version__)
@click.option('--num_threads', metavar='<threads>', default=1, type=int,
              show_default=True, callback=options.valid_int_gt_zero,
              help='Number of threads for OPENBLAS/MKL/OMP used in NumPy')
@click.option('--verbose', '-v', is_flag=True, help='Be verbose')
@click.option('--quiet', '-q', is_flag=True, help='Be quiet')
@click.pass_context
def cli(ctx, num_threads, verbose, quiet):
    # Logging config
    logger = logging.getLogger('yatsm')
    if verbose:
        logger.setLevel(logging.DEBUG)
    if quiet:
        logger.setLevel(logging.WARNING)

    # Set num_threads for NumPy linear algebra via MKL, OPENBLAS, or OPM
    for envvar in NP_THREAD_VARS:
        if envvar in os.environ:
            logger.warning('Overriding %s with %i' % (envvar, num_threads))
        os.environ[envvar] = str(num_threads)
