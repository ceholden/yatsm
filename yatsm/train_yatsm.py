#!/usr/bin/env python
""" Yet Another Timeseries Model (YATSM) - run script for classifier training

Usage: train_yatsm.py [options] <config_file> <roi_image> <output_model>

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

gdal.AllRegister()
gdal.UseExceptions()

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main(dataset_config, yatsm_config, roi, output):
    """ """
    pass


if __name__ == '__main__':
    # Arguments
    args = docopt(__doc__,
                  version=__version__)

    # Validate config file input
    config_file = args['<config_file>']
    if not os.path.isfile(args['<config_file>']):
        print('Error - <config_file> specified is not a file')
        sys.exit(1)

    # Validate input ROI
    roi = args['<roi_image>']
    try:
        ds = gdal.Open(roi, gdal.GA_ReadOnly)
    except:
        print('Error - cannot open <roi_image>')
        sys.exit(1)
    ds = None

    # Validate output model
    output = args['<output_model>']
    if os.path.isfile(output):
        print('Error - <output_model> already exists')
        sys.exit(1)
    if not os.path.exists(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except:
            print('Error - output directory does not exist \
                  and cannot be created')
            sys.exit(1)
    elif not os.access(os.path.dirname(output), os.W_OK):
        print('Error - cannot write to directory containing <output_model>')
        sys.exit(1)

    # Options
    mask = args['--mask'].replace(',', ' ').split(' ')
    try:
        mask = [int(m) for m in mask if m != '']
    except:
        print('Error - could not parse mask input option to list of integers')
        sys.exit(1)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--quiet']:
        logger.setLevel(logging.WARNING)

    # Read in config file
    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_config, yatsm_config = parse_config_file(config)

    main(dataset_config, yatsm_config, roi, output)
