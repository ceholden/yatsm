#!/usr/bin/env python
""" Cache data for Yet Another Timeseries Model (YATSM)

Usage:
    cache_yatsm.py [options] <config_file> <job_number> <total_jobs>

Options:
    --update=<pattern>          Create new cache files by updating old cache
                                    files matching provided pattern
    --interlace                 Assign rows
    -q --quiet                  Show only error messages
    -v --verbose                Show verbose messages
    --version                   Print program version and exit
    -h --help                   Show help

"""
from __future__ import division, print_function

import fnmatch
import logging
import os
import sys
import time

from docopt import docopt
import numpy as np

# Handle running as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.log_yatsm import logger
from yatsm import cache, config_parser, reader, utils


def main(args):
    # Parse and validate configuration file
    dataset_config, yatsm_config = config_parser.parse_config_file(
        args['config_file'])

    if not os.path.isdir(dataset_config['cache_line_dir']):
        os.makedirs(dataset_config['cache_line_dir'])

    dates, images = utils.csvfile_to_dataset(
        dataset_config['input_file'],
        date_format=dataset_config['date_format']
    )

    image_IDs = utils.get_image_IDs(images)

    nrow, ncol, nband, dtype = reader.get_image_attribute(images[0])

    # Determine lines to work on
    job_lines = utils.calculate_lines(args['job_number'],
                                      args['total_jobs'],
                                      nrow,
                                      interlaced=args['interlace'])
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Determine file reader
    if dataset_config['use_bip_reader']:
        logger.debug('Reading in data from disk using BIP reader')
        image_reader = reader.read_row_BIP
        image_reader_kwargs = {'size': (ncol, nband),
                               'dtype': dtype}
    else:
        logger.debug('Reading in data from disk using GDAL')
        image_reader = reader.read_row_GDAL
        image_reader_kwargs = {}

    # Attempt to update cache files
    previous_cache = None
    if args['update_pattern']:
        previous_cache = fnmatch.filter(
            os.listdir(dataset_config['cache_line_dir']),
            args['update_pattern'])

        if not previous_cache:
            logger.warning('Could not find cache files to update with pattern'
                           '{p}'.format(p=args['update_pattern']))
        else:
            logger.debug('Found {n} previously cached files to update'.format(
                n=len(previous_cache)))

    for job_line in job_lines:
        cache_filename = cache.get_line_cache_name(
            dataset_config, len(images), job_line, nband)
        logger.debug('Caching line {l} to {f}'.format(
            l=job_line, f=cache_filename))
        start_time = time.time()

        # Find matching cache file
        update = False
        if previous_cache:
            pattern = cache.get_line_cache_pattern(
                job_line, nband, regex=False)

            potential = fnmatch.filter(previous_cache, pattern)

            if not potential:
                logger.info('Found zero previous cache files for '
                            'line {l}'.format(l=job_line))
            elif len(potential) > 1:
                logger.info('Found more than one previous cache file for '
                            'line {l}. Keeping first'.format(l=job_line))
                update = os.path.join(dataset_config['cache_line_dir'],
                                      potential[0])
            else:
                update = os.path.join(dataset_config['cache_line_dir'],
                                      potential[0])

            logger.info('Updating from cache file {f}'.format(f=update))

        if update:
            cache.update_cache_file(
                images, image_IDs,
                update, cache_filename,
                job_line, image_reader, image_reader_kwargs
            )
        else:
            if dataset_config['use_bip_reader']:
                # Use BIP reader
                logger.debug('Reading in data from disk using BIP reader')
                Y = reader.read_row_BIP(images, job_line, (ncol, nband), dtype)
            else:
                # Read in data just using GDAL
                logger.debug('Reading in data from disk using GDAL')
                Y = reader.read_row_GDAL(images, job_line)
            cache.write_cache_file(cache_filename, Y, image_IDs)

        logger.debug('Took {s}s to cache the data'.format(
            s=round(time.time() - start_time, 2)))


def parse_args(args):
    """ Parse, format, and validate args from docopt """
    parsed = {}
    # Validate input arguments
    parsed['config_file'] = args['<config_file>']
    if not os.path.isfile(parsed['config_file']):
        print('Error - <config_file> specified is not a file')
        sys.exit(1)

    try:
        parsed['job_number'] = int(args['<job_number>'])
    except:
        print('Error - <job_number> must be an integer greater than 0')
        sys.exit(1)
    if parsed['job_number'] <= 0:
        print('Error - <job_number> cannot be less than or equal to 0')
        sys.exit(1)
    parsed['job_number'] -= 1

    try:
        parsed['total_jobs'] = int(args['<total_jobs>'])
    except:
        print('Error - <total_jobs> must be an integer')
        sys.exit(1)

    parsed['update_pattern'] = args['--update']

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--quiet']:
        logger.setLevel(logging.WARNING)

    parsed['interlace'] = args['--interlace']

    return parsed


if __name__ == '__main__':
    args = parse_args(docopt(__doc__, version=__version__))

    main(args)
