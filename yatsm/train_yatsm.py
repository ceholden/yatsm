#!/usr/bin/env python
""" Yet Another Timeseries Model (YATSM) - run script for classifier training

Usage: train_yatsm.py [options] <yatsm_config> <classifier_config>

Options:
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
from datetime import datetime as dt
from itertools import izip
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
import utils

gdal.AllRegister()
gdal.UseExceptions()

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class TrainingDataException(Exception):
    """ Custom exception for errors with training data """
    pass


def get_training_inputs(dataset_config, exit_on_missing=False):
    """ Returns X features and y labels specified in config file

    Args:
      dataset_config (dict): dataset configuration
      exit_on_missing (bool, optional): exit if input feature cannot be found

    Returns:
      X (np.ndarray): matrix of feature inputs for each training data sample
      y (np.ndarray): array of labeled training data samples

    """
    # Find and parse training data
    try:
        roi_ds = gdal.Open(dataset_config['training_image'], gdal.GA_ReadOnly)
    except:
        logger.error('Could not read in training image')
        raise
    logger.info('Reading in training data')
    roi = roi_ds.GetRasterBand(1).ReadAsArray()

    # Determine start and end dates of training sample relevance
    try:
        training_start = dt.strptime(
            dataset_config['training_start'],
            dataset_config['training_date_format']).toordinal()
        training_end = dt.strptime(
            dataset_config['training_end'],
            dataset_config['training_date_format']).toordinal()
    except:
        logger.error('Failed to parse training data start or end dates')
        raise

    # Loop through samples in ROI extracting features
    mask = np.logical_and.reduce([
        roi != mv for mv in dataset_config['mask_values']]).astype(np.uint8)
    row, col = np.where(mask)
    y = roi[row, col]

    X = []

    for _row, _col, _y in izip(y, row, col):
        # Load result
        rec = np.load(utils.get_output_name(dataset_config, _row))['record']
        # Find intersecting time segment
        i = np.where((rec['start'] < training_start) &
                     (rec['end'] > training_end) &
                     (rec['px'] == _col))[0]

        if i.size == 0:
            txt = ('Could not find model for label {l} at x/y {c}/{r}'.format(
                l=_y, c=_col, r=_row))
            if not exit_on_missing:
                logger.debug(txt)
                continue
            else:
                raise Exception(txt)
        elif i.size > 1:
            raise TrainingDataException('Found more than one valid model for \
                label {l} at x/y {c}/{r}'.format(l=_y, x=_col, y=_row))

        # Extract coefficients with intercept term rescaled
        coef = rec[i]['coef']
        coef[0, :] = (coef[0, :] +
                      coef[1, :] * (rec[i]['start'] + rec[i]['end']) / 2.0)

        X.append(np.concatenate(coef.reshape(coef.size),
                                rec[i]['rmse'][0]))

    return (np.array(X), y)

def main(dataset_config, yatsm_config, algo):
    """ YATSM trainining main

    Args:
      dataset_config (dict): options for the dataset
      yatsm_config (dict): options for the change detection algorithm
      algo (sklearn classifier): classification algorithm helper class


    """
    # Cache file for training data
    has_cache = False
    if dataset_config['cache_Xy']:
        # If doesn't exist, retrieve it
        if not os.path.isfile(dataset_config['cache_Xy']):
            logger.info('Could not retrieve cache file for Xy')
            logger.info('    file: {f}'.format(dataset_config['cache_Xy']))
        else:
            logger.info('Restoring X/y from cache file')
            has_cache = True

    if not has_cache:
        logger.info('Reading in X/y')
        X, y = get_training_inputs(dataset_config)
        logger.info('Read in X/y')
    else:
        # TODO
        logger.info('Reading in from cache file {f}'.format(
            f=dataset_config['cache_Xy']))


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

    if not dataset_config['training_image'] or \
            not os.path.isfile(dataset_config['training_image']):
        logger.error('Training data image {f} does not exist'.format(
            f=dataset_config['training_image']))
        sys.exit(1)

    # Parse classifier config
    algorithm_helper = classifiers.ini_to_algorthm(classifier_config_file)

    main(dataset_config, yatsm_config, algorithm_helper,)
