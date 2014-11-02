""" Module storing classifiers for YATSM

Contains utilities and helper classes for classifying timeseries generated
using YATSM change detection.
"""
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import logging

from sklearn_helper import RandomForestHelper

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
