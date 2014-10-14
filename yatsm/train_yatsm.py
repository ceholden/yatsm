#!/usr/bin/env python
""" Yet Another Timeseries Model (YATSM) - run script for classifier training

Usage: train_yatsm.py [options] <yatsm_config> <classifier_config>

Options:
    --mask=<values>             Values to mask in <roi_image> [default: 0]
    -v --verbose                Show verbose debugging messages
    -q --quiet                  Show only error messages
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
import time

from docopt import docopt
import numpy as np
from osgeo import gdal

from version import __version__
from config_parser import parse_config_file
import classifiers

gdal.AllRegister()
gdal.UseExceptions()

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main(dataset_config, algo):
    """ """
    pass


if __name__ == '__main__':
    # Arguments
    args = docopt(__doc__,
                  version=__version__)

    # Validate dataset config file input
    yatsm_config_file = args['<yatsm_config>']
    if not os.path.isfile(yatsm_config_file):
        print('Error - <yatsm_config> specified is not a file')
        sys.exit(1)

    # Validate classifier config file input
    classifier_config_file = args['<classifier_config>']
    if not os.path.isfile(classifier_config_file):
        print('Error - <classifier_config> specified is not a file')
        sys.exit(1)

    # Options
    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--quiet']:
        logger.setLevel(logging.WARNING)

    # Parse YATSM config
    yatsm_config = configparser.ConfigParser()
    yatsm_config.read(yatsm_config_file)
    dataset_config, yatsm_config = parse_config_file(yatsm_config)

    # Parse classifier config
    algorithm_helper = classifiers.ini_to_algorthm(classifier_config_file)

    main(dataset_config, algorithm_helper)
