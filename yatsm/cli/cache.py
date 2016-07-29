""" Command line interface for working with cached data for YATSM algorithms
"""
import fnmatch
import logging
import os
import time

import click

from . import options
from .. import io


logger = logging.getLogger('yatsm')


@click.command(short_help='Create or update cached timeseries data for YATSM')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.option('--update', 'update_pattern', metavar='<pattern>',
              help='Create new cache files by updating old cache files '
                   'matching provided pattern')
@click.option('--interlace', is_flag=True,
              help='Assign rows interlaced by job instead of sequentially')
@click.pass_context
def cache(ctx, config, job_number, total_jobs, update_pattern, interlace):
    from ..cache import (get_line_cache_name, get_line_cache_pattern,
                         update_cache_file, write_cache_file)
    from ..config_parser import parse_config_file
    from ..utils import csvfile_to_dataframe, distribute_jobs, get_image_IDs

    cfg = parse_config_file(config)
    if not os.path.isdir(cfg['dataset']['cache_line_dir']):
        os.makedirs(cfg['dataset']['cache_line_dir'])

    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              cfg['dataset']['date_format'])
    df['image_IDs'] = get_image_IDs(df['filename'])

    nrow, ncol, nband, dtype = io.get_image_attribute(df['filename'][0])

    # Determine lines to work on
    job_lines = distribute_jobs(job_number, total_jobs, nrow,
                                interlaced=interlace)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Determine file reader
    if cfg['dataset']['use_bip_reader']:
        logger.debug('Reading in data from disk using BIP reader')
        image_reader = io.bip_reader
    else:
        logger.debug('Reading in data from disk using GDAL')
        image_reader = io.gdal_reader

    # Attempt to update cache files
    previous_cache = None
    if update_pattern:
        previous_cache = fnmatch.filter(
            os.listdir(cfg['dataset']['cache_line_dir']), update_pattern)

        if not previous_cache:
            logger.warning('Could not find cache files to update with pattern '
                           '%s' % update_pattern)
        else:
            logger.debug('Found %s previously cached files to update' %
                         len(previous_cache))

    for job_line in job_lines:
        cache_filename = get_line_cache_name(cfg['dataset'], len(df),
                                             job_line, nband)
        logger.debug('Caching line {l} to {f}'.format(
            l=job_line, f=cache_filename))
        start_time = time.time()

        # Find matching cache file
        update = False
        if previous_cache:
            pattern = get_line_cache_pattern(job_line, nband, regex=False)

            potential = fnmatch.filter(previous_cache, pattern)

            if not potential:
                logger.info('Found zero previous cache files for '
                            'line {l}'.format(l=job_line))
            elif len(potential) > 1:
                logger.info('Found more than one previous cache file for '
                            'line {l}. Keeping first'.format(l=job_line))
                update = os.path.join(cfg['dataset']['cache_line_dir'],
                                      potential[0])
            else:
                update = os.path.join(cfg['dataset']['cache_line_dir'],
                                      potential[0])

            logger.info('Updating from cache file {f}'.format(f=update))

        if update:
            update_cache_file(df['filename'], df['image_IDs'],
                              update, cache_filename,
                              job_line, image_reader)
        else:
            if cfg['dataset']['use_bip_reader']:
                # Use BIP reader
                logger.debug('Reading in data from disk using BIP reader')
                Y = io.bip_reader.read_row(df['filename'], job_line)
            else:
                # Read in data just using GDAL
                logger.debug('Reading in data from disk using GDAL')
                Y = io.gdal_reader.read_row(df['filename'], job_line)
            write_cache_file(cache_filename, Y, df['image_IDs'])

        logger.debug('Took {s}s to cache the data'.format(
            s=round(time.time() - start_time, 2)))
