""" Command line interface for running YATSM on image segments """
import logging
import sys

import click

import yatsm
from yatsm.cli.cli import (
    cli,
    config_file_arg, job_number_arg, total_jobs_arg,
    format_opt, rootdir_opt, resultdir_opt, exampleimg_opt
)

logger = logging.getLogger('yatsm')


@cli.command(short_help='Run YATSM on a segmented image')
@config_file_arg
@job_number_arg
@total_jobs_arg
@rootdir_opt
@resultdir_opt
@exampleimg_opt
@format_opt
@click.pass_context
def segment(ctx, config, job_number, total_jobs,
            root, result, image, format):
    logger.debug('TODO')
