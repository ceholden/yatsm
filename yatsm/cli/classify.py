""" Command line interface for classifying YATSM algorithm output
"""
from __future__ import division

import logging
import os
import time

import click

from . import options

logger = logging.getLogger('yatsm')


@click.command(short_help='Classify entire images using trained algorithm')
@options.arg_config_file
@click.argument('algo', metavar='<trained algorithm>',
                type=click.Path(readable=True, resolve_path=True))
@options.arg_job_number
@options.arg_total_jobs
@click.option('--resume', is_flag=True,
              help="Resume classification (don't overwrite)")
@click.pass_context
def classify(ctx, configfile, algo, job_number, total_jobs, resume):
    from sklearn.externals import joblib

    from ..classification.batch import classify_line, try_resume
    from ..config_parser import parse_config_file
    from ..io import get_image_attribute
    from ..utils import distribute_jobs, get_output_name, csvfile_to_dataframe

    cfg = parse_config_file(configfile)

    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              cfg['dataset']['date_format'])
    nrow = get_image_attribute(df['filename'][0])[0]

    classifier = joblib.load(algo)

    # Split into lines and classify
    job_lines = distribute_jobs(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    start_time = time.time()
    logger.info('Starting to run lines')
    for job_line in job_lines:
        filename = get_output_name(cfg['dataset'], job_line)
        if not os.path.exists(filename):
            logger.warning('No model result found for line {l} '
                           '(file {f})'.format(l=job_line, f=filename))
            pass

        if resume and try_resume(filename):
            logger.debug('Already processed line {l}'.format(l=job_line))
            continue

        logger.debug('Classifying line {l}'.format(l=job_line))
        classify_line(filename, classifier)

    logger.debug('Completed {n} lines in {m} minutes'.format(
        n=len(job_lines),
        m=round((time.time() - start_time) / 60.0, 2))
    )
