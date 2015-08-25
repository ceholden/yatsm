import inspect
import StringIO

import numpy as np
import sklearn.linear_model
import sklearn.externals.joblib
import yaml

from . import algorithms
from .log_yatsm import logger
from .version import __version__


def parse_config_file(config_file):
    """ Parse YAML config file

    Args:
        config_file (str): path to YAML config file

    Returns:
        dict: dict of sub-dicts, each sub-dict containing configuration keys
            and values pertinent to a process or algorithm. Pickled `sklearn`
            models will be loaded and returned as an object within the dict

    Raises:
        KeyError: raise KeyError if configuration file is not specified
            correctly

    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    # Ensure algorithm & prediction sections are specified
    if 'YATSM' not in cfg:
        raise KeyError('YATSM must be a section in configuration YAML file')

    if 'algorithm' not in cfg['YATSM']:
        raise KeyError('YATSM section does not declare an algorithm')
    algo = cfg['YATSM']['algorithm']
    if algo not in cfg:
        raise KeyError('Algorithm specified (%s) is not parameterized in '
                       'configuration file' % algo)

    if 'prediction' not in cfg['YATSM']:
        raise KeyError('YATSM section does not declare a prediction method')
    if cfg['YATSM']['prediction'] not in cfg:
        raise KeyError('Prediction method specified (%s) is not parameterized '
                       'in configuration file' % cfg['YATSM']['prediction'])

    # Embed algorithm in YATSM key
    if algo not in algorithms.available:
        raise NotImplementedError('Algorithm specified (%s) is not currently '
                                  'available' % algo)
    cfg['YATSM']['algorithm_cls'] = getattr(algorithms, algo)
    if not cfg['YATSM']['algorithm_cls']:
        raise KeyError('Could not find algorithm specified (%s) in '
                       '`yatsm.algorithms`' % algo)

    # Expand min/max values to all bands
    n_bands = cfg['dataset']['n_bands']
    mins, maxes = cfg['dataset']['min_values'], cfg['dataset']['max_values']
    if isinstance(mins, (float, int)):
        cfg['dataset']['min_values'] = np.asarray([mins] * n_bands)
    if isinstance(maxes, (float, int)):
        cfg['dataset']['max_values'] = np.asarray([maxes] * n_bands)

    # Add in dummy phenology and classification dicts if not included
    if 'phenology' not in cfg:
        cfg['phenology'] = {'enable': False}

    if 'classification' not in cfg:
        cfg['classification'] = {'training_image': None}

    # Load sklearn objects
    if cfg['YATSM']['prediction'] in dir(sklearn.linear_model):
        reg = sklearn.externals.joblib.load(
            cfg[cfg['YATSM']['prediction']]['pickle'])
        cfg['YATSM']['prediction'] = reg
    else:
        raise KeyError('Currently cannot unpickle prediction objects not '
                       'in sklearn')

    return cfg
