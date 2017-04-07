""" Loads all commands for YATSM command line interface

Modeled after very nice `click` interface for `rasterio`:
https://github.com/mapbox/rasterio/blob/master/rasterio/rio/main.py

"""
from __future__ import absolute_import

import logging
import os
from pkg_resources import iter_entry_points
import sys

import click
import click_plugins
import cligj


# NumPy/etc linear algebra multithreading related variables
NP_THREAD_VARS = [
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPM_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_NUM_THREADS'
]
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
from ._logger import config_logging


# YATSM CLI group
_context = dict(
    token_normalize_func=lambda x: x.lower(),
    help_option_names=['--help', '-h'],
    auto_envvar_prefix='YATSM'
)

@click_plugins.with_plugins(ep for ep in
                            iter_entry_points('yatsm.cli'))
@click.group(context_settings=_context)
@options.opt_config
@click.version_option(yatsm.__version__)
@click.option('--num_threads', default=1, type=int,
              show_default=True, callback=options.valid_int_gt_zero,
              help='Number of threads for OPENBLAS/MKL/OMP used in NumPy')
@cligj.verbose_opt
@cligj.quiet_opt
@click.pass_context
def cli(ctx, config, num_threads, verbose, quiet):
    """ YATSM Applications

    Note:
    Be sure to specify the configuration file you want to work with as
    ``yatsm -C CONFIG_FILE [COMMAND]``. You might also consider defining
    the path to the configuration file as an environment variable,
    ``YATSM_CONFIG``. Defining this variable allows YATSM to retrieve
    the location of your configuration file without the need to explicitly
    include the ``-C`` when you write ``yatsm [COMMAND...]`` commands,
    simplifying interactive use.
    """
    verbosity = verbose - quiet
    level = max(10, 30 - 10 * verbosity)
    logger = config_logging(level, config=None)  # TODO: dictConfig file arg
    ctx.obj = ctx.obj or {}
    ctx.obj['verbosity'] = verbosity
    ctx.obj['config'] = config
