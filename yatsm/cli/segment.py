""" Command line interface for running YATSM on image segments """
import logging
import sys

import click

from yatsm.cli.cli import (
    cli,
    config_file_arg, job_number_arg, total_jobs_arg,
    format_opt, rootdir_opt, resultdir_opt, exampleimg_opt
)
import yatsm.config_parser
import yatsm.reader
import yatsm.segment
import yatsm.utils

logger = logging.getLogger('yatsm')


@cli.command(short_help='Run YATSM on a segmented image')
@config_file_arg
@job_number_arg
@total_jobs_arg
@click.pass_context
def segment(ctx, config, job_number, total_jobs):
    # Parse config
    dataset_config, yatsm_config = \
        yatsm.config_parser.parse_config_file(config)

    # Read in segmentation image
    if not yatsm_config['segmentation']:
        logger.error('No segmentation image specified in configuration file.')
        sys.exit(1)
    segment = yatsm.reader.read_image(yatsm_config['segmentation'])[0]

    # Calculate segments for this job
    n_segment = segment.max()
    job_segments = yatsm.utils.calculate_lines(job_number, total_jobs,
                                               n_segment, interlaced=False)

    # What lines are required?
    job_lines = yatsm.segment.segments_to_lines(segment, job_segments)

    # Read and store all required lines

