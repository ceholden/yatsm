""" Command line interface for running YATSM on image lines """
import logging
import sys

import click

import yatsm
from yatsm.cli.cli import (
    cli,
    config_file_arg, job_number_arg, total_jobs_arg
)

logger = logging.getLogger('yatsm')


@cli.command(short_help='Run YATSM on an entire image line by line')
@config_file_arg
@job_number_arg
@total_jobs_arg
@click.option('--check', is_flag=True,
              help='Check that images exist')
@click.option('--check_cache', is_flag=True,
              help='Check that cache file contains matching data')
@click.option('--do-not-run', is_flag=True,
              help='Do not run YATSM (useful for just caching data)')
@click.option('--verbose-yatsm', is_flag=True,
              help='Show verbose debugging messages in YATSM')
@click.pass_context
def line(ctx, config, job_number, total_jobs,
         do_not_run, check, check_cache, verbose_yatsm):
    logger.debug('TODO')

