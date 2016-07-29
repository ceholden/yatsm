""" Module storing classifiers for YATSM

Contains utilities and helper classes for classifying timeseries generated
using YATSM change detection.
"""
import logging

from sklearn.ensemble import RandomForestClassifier
import yaml

from ..errors import AlgorithmNotFoundException

logger = logging.getLogger('yatsm')

_algorithms = {
    'RandomForest': RandomForestClassifier
}


def cfg_to_algorithm(config_file):
    """ Return instance of classification algorithm helper from config file

    Args:
        config_file (str): location of configuration file for algorithm

    Returns:
        tuple: scikit-learn estimator (object) and configuration file (dict)

    Raises:
        KeyError: raise if configuration file is malformed
        AlgorithmNotFoundException: raise if algorithm is not implemented in
            YATSM
        TypeError: raise if configuration file cannot be used to initialize
            the classifier

    """
    # Determine which algorithm is used
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error('Could not read config file {} ({})'
                     .format(config_file, str(e)))
        raise

    algo_name = config['algorithm']
    if algo_name not in _algorithms.keys():
        raise AlgorithmNotFoundException(
            'Could not process unknown algorithm named "%s"' % algo_name)
    else:
        algo = _algorithms[algo_name]

    if algo_name not in config:
        logger.warning('%s algorithm parameters not found in config file %s. '
                       'Using default values.' % (algo_name, config_file))
        config[algo_name] = {}

    # Try to load algorithm using hyperparameters from config
    try:
        sklearn_algo = algo(**config[algo_name].get('init', {}))
    except TypeError:
        logger.error('Cannot initialize %s classifier. Config file %s '
                     'contains unknown options' % (algo_name, config_file))
        raise

    return sklearn_algo, config
