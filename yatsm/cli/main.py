""" Loads all commands for YATSM command line interface

Modeled after very nice `click` interface for `rasterio`:
https://github.com/mapbox/rasterio/blob/master/rasterio/rio/main.py

"""
import logging
import os
from pkg_resources import iter_entry_points
import sys

import click
import click_plugins

logger = logging.getLogger('yatsm')
logger_algo = logging.getLogger('yatsm_algo')

# NumPy linear algebra multithreading related variables
NP_THREAD_VARS = ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OPM_NUM_THREADS']


def set_np_thread_vars(n):
    for envvar in NP_THREAD_VARS:
        if envvar in os.environ:
            logger.warning('Overriding %s with --num_threads=%i'
                           % (envvar, n))
        os.environ[envvar] = str(n)

# If --num_threads set, parse it before click CLI interface so envvars are
# set BEFORE numpy is imported
if '--num_threads' in sys.argv:
    n_threads = sys.argv[sys.argv.index('--num_threads') + 1]
    try:
        n_threads = int(n_threads)
    except ValueError as e:
        raise click.BadParameter('Cannot parse <threads> to an integer '
                                 '(--num_threads=%s): %s' %
                                 (n_threads, e.message))
    else:
        set_np_thread_vars(n_threads)
else:
    # Default to 1
    set_np_thread_vars(1)

# Resume YATSM imports after NumPy has been configured
import yatsm  # flake8: noqa
from . import options  # flake8: noqa

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
@click.option('--verbose-yatsm', is_flag=True,
              help='Show verbose debugging messages in YATSM algorithm')
@click.option('--quiet', '-q', is_flag=True, help='Be quiet')
@click.pass_context
def cli(ctx, num_threads, verbose, verbose_yatsm, quiet):
    # Logging config
    if verbose:
        logger.setLevel(logging.DEBUG)
    if verbose_yatsm:
        logger_algo.setLevel(logging.DEBUG)
    if quiet:
        logger.setLevel(logging.WARNING)
        logger_algo.setLevel(logging.WARNING)
