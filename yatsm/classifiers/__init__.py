""" Module storing classifiers for YATSM

Contains utilities and helper classes for classifying timeseries generated
using YATSM change detection.
"""
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import logging

import numpy as np
import scipy.ndimage

from random_forest_helper import RandomForestHelper

logger = logging.getLogger('yatsm')

_algorithms = {
    'RandomForest': RandomForestHelper
}


class AlgorithmNotFoundException(Exception):
    """ Custom exception for algorithm config files without handlers """
    pass


def ini_to_algorthm(config_file):
    """ Return instance of classification algorithm helper from config file

    Args:
      config_file (str): location of configuration file for algorithm

    """
    # Determine which algorithm is used
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
    except:
        logger.error('Could not read config file {f}'.format(f=config_file))
        raise

    algo_name = config.get('metadata', 'Algorithm')

    if algo_name not in _algorithms.keys():
        raise AlgorithmNotFoundException(
            'Could not process algorithm named "{n}"'.format(n=algo_name))
    else:
        algo = _algorithms[algo_name]

    # Re-read using defaults
    config = configparser.ConfigParser(defaults=algo.defaults)
    config.read(config_file)

    # Return instance of algorithm
    return algo(config)


def spatial_crossvalidate(roi, mask_values=[0], n_folds=3):
    """ Return crossvalidated accuracy for spatial training data

    Training data samples physically located next to test samples are likely to
    be strongly related due to spatial autocorrelation. This violation of
    independence will artificially inflate crossvalidated measures of
    algorithm performance. This function will performance crossvalidation by
    treating contiguous training data samples as one sample during
    crossvalidation.

    Args:
      roi (np.ndarray): training data raster to sample from
      mask_values (np.ndarray, optional): mask values to ignore from raster
      n_folds (int, optional): number of folds to use

    Returns:

    """
    if isinstance(mask_values, int):
        mask_values = [mask_values]

    mask = np.logical_or.reduce([roi == mv for mv in mask_values])

    labeled, n_labels = scipy.ndimage.label(roi)

    labels = np.unique(labeled)
    unmasked_labels = np.in1d(labels, np.unique(labeled[~mask]))
    labels = labels[unmasked_labels]









