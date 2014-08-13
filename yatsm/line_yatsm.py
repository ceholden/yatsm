""" Yet Another Timeseries Model (YATSM) - run script for lines of images

Usage: line_yatsm.py [options] <config_file> <job_number> <total_jobs>

Options:
    -v --verbose                Show verbose debugging messages
    -h --help                   Show help

"""
from __future__ import division, print_function

# Support namechange in Python3
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import logging
import os
import sys

from docopt import docopt

from osgeo import gdal


def _check_config_file(config):
    """ Checks config file for required fields """
    pass

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
        print('Error - <job_number> must be an integer')
        sys.exit(1)

    try:
        total_jobs = int(args['<total_jobs>'])
    except:
        print('Error - <total_jobs> must be an integer')
        sys.exit(1)

    # Setup logger
    if args['--verbose']:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                        level=loglevel,
                        datefmt='%H:%M:%S')

    # Parse and validate configuration file
    config = configparser.ConfigParser()
    config.read(config_file)


