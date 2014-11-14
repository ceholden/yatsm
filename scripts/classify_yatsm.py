#!/usr/bin/env python
""" Yet Another Timeseris Model (YATSM) - classify with trained algorithm

Usage:
    classify_yatsm.py [options] <config_file> <trained_algorithm>
        <job_number> <total_jobs>

Options:
    --verbose                   Show verbose debugging messages
    --version                   Print program version and exit
    -h --help                   Show help and exit

"""
from __future__ import division, print_function

import logging
import os
import pickle
import sys

from docopt import docopt

import numpy as numpy

# Handle runnin as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.config_parser import parse_config_file
from yatsm.utils import (calculate_lines, get_output_name,
                         csvfile_to_dataset, get_image_attribute)


FORMAT = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')


def parse_args(args):
    """ Returns dictionary of parsed and validated command arguments

    Args:
      args (dict): Arguments from user

    Returns:
      dict: Parsed and validated arguments

    """
    parsed_args = {}
    # Required args
    parsed_args['config_file'] = args['<config_file>']

    parsed_args['algo'] = args['<trained_algorithm>']

    try:
        job_number = int(args['<job_number>'])
    except:
        print('Error - <job_number> must be an integer greater than 0')
        sys.exit(1)
    if job_number <= 0:
        print('Error - <job_number> cannot be less than or equal to 0')
        sys.exit(1)
    parsed_args['job_number'] = job_number

    try:
        parsed_args['total_jobs'] = int(args['<total_jobs>'])
    except:
        print('Error - <total_jobs> must be an integer')
        sys.exit(1)
    if parsed_args['job_number'] > parsed_args['total_jobs']:
        print('Error - <job_number> must be less than or equal to total jobs')
        sys.exit(1)

    return parsed_args


def main(args):
    """ Classify dataset """
    # Parse config and file data
    dataset_config, yatsm_config = parse_config_file(args['config_file'])

    dates, images = csvfile_to_dataset(
        dataset_config['input_file'],
        date_format=dataset_config['date_format']
    )
    nrow, _, _, _ = get_image_attribute(images[0])

    job_lines = calculate_lines(args['job_number'] - 1, args['total_jobs'],
                                nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))




if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)

    args = parse_args(args)
    main(args)
