#!/usr/bin/env python
""" Yet Another Timeseries Model (YATSM) - run script for lines of images

Usage: line_yatsm.py [options] <config_file> <job_number> <total_jobs>

Options:
    --check                     Check that images exist
    -v --verbose                Show verbose debugging messages
    -h --help                   Show help

"""
from __future__ import division, print_function

# Support namechange in Python3
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import csv
from datetime import datetime as dt
import logging
import os
import sys

from docopt import docopt

import numpy as np
from osgeo import gdal


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


# IMAGE DATASET READING
def find_images(input_file, date_format='%Y-%j'):
    """ Return sorted filenames of images from input text file

    Args:
      input_file (string)       Text file of dates and files
      date_format (string)      Format of dates in file

    Returns:
      (ndarray, ndarray)        Paired dates and filenames of stacked images

    """
    # Store index of date and image
    i_date = 0
    i_image = 1

    dates = []
    images = []

    logger.debug('Opening image dataset file')
    with open(input_file, 'rb') as f:
        reader = csv.reader(f)

        # Figure out which index is for what
        row = reader.next()

        try:
            dt.strptime(row[i_date], date_format).toordinal()
        except:
            logger.debug('Could not parse first column to ordinal date')

            try:
                dt.strptime(row[i_image], date_format).toordinal()
            except:
                logger.debug('Could not parse second column to ordinal date')
                logger.error('Could not parse any columns to ordinal date')
                logger.error('Input config file: {f}'.format(f=config))
                logger.error('Input dataset file: {f}'.format(f=input_file))
                logger.error('Date format: {f}'.format(f=date_format))
                raise
                sys.exit(1)
            else:
                i_date = 1
                i_image = 0

        f.seek(0)

        logger.debug('Reading in image date and filenames')
        for row in reader:
            dates.append(dt.strptime(row[i_date], date_format).toordinal())
            images.append(row[i_image])

        return (np.array(dates), np.array(images))


# CONFIG FILE PARSING
def _parse_config_v_zero_pt_one(config):
    """ Parses config file for version 0.1.x """

    dataset_config = dict.fromkeys(['input_file', 'date_format',
                                    'output',
                                    'n_bands', 'mask_band',
                                    'green_band', 'swir1_band',
                                    'use_bip_reader'])

    for k in dataset_config:
        dataset_config[k] = config.get('dataset', k)

    yatsm_config = dict.fromkeys(['consecutive', 'threshold', 'min_obs',
                                  'freq', 'lassocv', 'reverse'])

    return (dataset_config, yatsm_config)


def parse_config_file(config):
    """ Parses config file into dictionary of attributes """

    dataset_config = None
    yatsm_config = None

    # Parse different versions
    version = config.get('metadata', 'version').split('.')

    # 0.1.x
    if version[0] == '0' and version[1] == '1':
        dataset_config, yatsm_config = _parse_config_v_zero_pt_one(config)

    return (dataset_config, yatsm_config)


def main(dataset_config, yatsm_config, check=False):
    """ Read in dataset and YATSM for a complete line """
    print(dataset_config)
    print(yatsm_config)

    # Read in dataset
    dates, images = find_images(dataset_config['input_file'])

    # Check for existence of files and remove missing
    if check:
        to_delete = []
        for i, img in enumerate(images):
            if not os.path.isfile(img):
                logger.warning('Could not find file {f} -- removing'.
                    format(f=img))
                to_delete.append(i)

        if len(to_delete) == 0:
            logger.debug('Checked and found all input images')
        else:
            logger.warning('Removing {n} images'.format(n=len(to_delete)))
            dates = np.delete(dates, np.array(to_delete))
            images = np.delete(images, np.array(to_delete))


if __name__ == '__main__':
    # Get arguments
    args = docopt(__doc__)

    # Validate input arguments
    config_file = args['<config_file>']
    if not os.path.isfile(args['<config_file>']):
        print('Error - <config_file> specified is not a file')
        sys.exit(1)

    try:
        job_number = int(args['<job_number>'])
    except:
        print('Error - <job_number> must be an integer greater than 0')
        sys.exit(1)
    if job_number <= 0:
        print('Error - <job_number> cannot be less than or equal to 0')
        sys.exit(1)
    job_number -= 1

    try:
        total_jobs = int(args['<total_jobs>'])
    except:
        print('Error - <total_jobs> must be an integer')
        sys.exit(1)

    # Check for existence of images?
    check = args['--check']

    # Setup logger
    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.DEBUG)

    # Parse and validate configuration file
    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_config, yatsm_config = parse_config_file(config)

    # Run YATSM
    main(dataset_config, yatsm_config, check=check)
