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

# Handle runnin as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.config_parser import parse_config_file
from yatsm import classifiers
from yatsm import utils

gdal.AllRegister()
gdal.UseExceptions()

logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')


class TrainingDataException(Exception):
    """ Custom exception for errors with training data """
    pass


def is_cache_old(cache_file, training_file):
    """ Indicates if cache file is older than training data file

    Args:
      cache_file (str): filename of the cache file
      training_file (str): filename of the training data file_

    Returns:
      old (bool): True if the cache file is older than the training data file
        and needs to be updated; False otherwise

    """
    if cache_file and os.path.isfile(cache_file):
        return os.stat(cache_file).st_mtime < os.stat(training_file).st_mtime
    else:
        return False


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
    out_y = []

    for _row, _col, _y in izip(row, col, y):
        # Load result
        try:
            rec = np.load(utils.get_output_name(dataset_config, _row))['record']
        except:
            logger.error('Could not open saved result file {f}'.format(
                f=utils.get_output_name(dataset_config, _row)))
            if exit_on_missing:
                raise
            else:
                continue
        # Find intersecting time segment
        i = np.where((rec['start'] < training_start) &
                     (rec['end'] > training_end) &
                     (rec['px'] == _col))[0]

        if i.size == 0:
            logger.debug(
                'Could not find model for label {l} at x/y {c}/{r}'.format(
                    l=_y, c=_col, r=_row))
            continue
        elif i.size > 1:
            raise TrainingDataException('Found more than one valid model for \
                label {l} at x/y {c}/{r}'.format(l=_y, x=_col, y=_row))

        # Extract coefficients with intercept term rescaled
        coef = rec[i]['coef'][0]
        coef[0, :] = (coef[0, :] +
                      coef[1, :] * (rec[i]['start'] + rec[i]['end']) / 2.0)

        X.append(np.concatenate(
            (coef.reshape(coef.size), rec[i]['rmse'][0])))
        out_y.append(_y)

    logger.info('Found matching time segments for {m} out of {n} labels'.
                format(m=len(out_y), n=y.size))

    return (np.array(X), np.array(out_y))


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
            logger.info('    file: {f}'.format(f=dataset_config['cache_Xy']))
        else:
            logger.info('Restoring X/y from cache file')
            has_cache = True

    # Check if we need to regenerate the cache file because training data is
    #   newer than the cache
    regenerate_cache = is_cache_old(dataset_config['cache_Xy'],
                                    dataset_config['training_image'])
    if regenerate_cache:
        logger.warning('Existing cache file older than training data ROI')
        logger.warning('Regenerating cache file')

    if not has_cache or regenerate_cache:
        logger.debug('Reading in X/y')
        X, y = get_training_inputs(dataset_config)
        logger.debug('Done reading in X/y')
    else:
        logger.debug('Reading in X/y from cache file {f}'.format(
            f=dataset_config['cache_Xy']))
        with np.load(dataset_config['cache_Xy']) as f:
            X = f['X']
            y = f['y']
        logger.debug('Read in X/y from cache file {f}'.format(
            f=dataset_config['cache_Xy']))

    # If cache didn't exist but is specified, create it for first time
    if not has_cache and dataset_config['cache_Xy']:
        logger.info('Saving X/y to cache file {f}'.format(
            f=dataset_config['cache_Xy']))
        try:
            np.savez(dataset_config['cache_Xy'], X=X, y=y)
        except:
            logger.error('Could not save X/y to cache file')
            raise

    # Do modeling
    logger.info('Training classifier')
    algo.fit(X, y)

    logger.info('Score on X/y: {p}'.format(p=algo.score(X, y)))
    logger.info('OOB score: {p}'.format(p=algo.oob_score_))

    # algo.predict_proba(X).sum(axis=1)

    from IPython.core.debugger import Pdb
    Pdb().set_trace()


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
    dataset_config, yatsm_config = parse_config_file(yatsm_config_file)

    if not dataset_config['training_image'] or \
            not os.path.isfile(dataset_config['training_image']):
        logger.error('Training data image {f} does not exist'.format(
            f=dataset_config['training_image']))
        sys.exit(1)

    # Parse classifier config
    algorithm_helper = classifiers.ini_to_algorthm(classifier_config_file)

    main(dataset_config, yatsm_config, algorithm_helper,)
