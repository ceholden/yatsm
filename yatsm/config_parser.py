import logging
import os

import numpy as np
import six
import sklearn.externals.joblib as joblib
import yaml

from . import algorithms
from .config import expand_envvars
from .regression.packaged import find_packaged_regressor, packaged_regressions

logger = logging.getLogger('yatsm')


def parse_config_file(config_file):
    """ Parse YAML config file

    Args:
        config_file (str): path to YAML config file

    Returns:
        dict: dict of sub-dicts, each sub-dict containing configuration keys
            and values pertinent to a process or algorithm. Pickled
            estimators compatible with ``scikit-learn``
            (i.e., that follow :class:`sklearn.base.BaseEstimator`)
            models will be loaded and returned as an object within the dict

    Raises:
        KeyError: raise KeyError if configuration file is not specified
            correctly

    """
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
    cfg = expand_envvars(cfg)

    # Ensure algorithm & prediction sections are specified
    if 'YATSM' not in cfg:
        raise KeyError('YATSM must be a section in configuration YAML file')
    if 'prediction' not in cfg['YATSM']:
        raise KeyError('YATSM section does not declare a prediction method')
    if 'algorithm' not in cfg['YATSM']:
        raise KeyError('YATSM section does not declare an algorithm')
    algo = cfg['YATSM']['algorithm']
    if algo not in cfg:
        raise KeyError('Algorithm specified (%s) is not parameterized in '
                       'configuration file' % algo)

    # Embed algorithm in YATSM key
    # TODO: broaden this concept to at least algo['change']
    if algo not in algorithms.available['change']:
        raise NotImplementedError('Algorithm specified (%s) is not currently '
                                  'available' % algo)
    cfg['YATSM']['algorithm_cls'] = algorithms.available['change'][algo]
    if not cfg['YATSM']['algorithm_cls']:
        raise KeyError('Could not find algorithm specified (%s) in '
                       '`yatsm.algorithms.available`' % algo)

    # Add in dummy phenology and classification dicts if not included
    if 'phenology' not in cfg:
        cfg['phenology'] = {'enable': False}

    if 'classification' not in cfg:
        cfg['classification'] = {'training_image': None}

    return convert_config(cfg)


def convert_config(cfg):
    """ Convert some configuration values to different values

    Args:
        cfg (dict): dict of sub-dicts, each sub-dict containing configuration
            keys and values pertinent to a process or algorithm

    Returns:
        dict: configuration dict with some items converted to different objects

    Raises:
        KeyError: raise KeyError if configuration file is not specified
            correctly
    """
    # Parse dataset:
    cfg = _parse_dataset_config(cfg)
    # Parse YATSM:
    cfg = _parse_YATSM_config(cfg)

    return cfg


def _parse_dataset_config(cfg):
    """ Parse "dataset:" configuration section
    """
    # Expand min/max values to all bands
    n_bands = cfg['dataset']['n_bands']
    mins, maxes = cfg['dataset']['min_values'], cfg['dataset']['max_values']
    if isinstance(mins, (float, int)):
        cfg['dataset']['min_values'] = np.asarray([mins] * n_bands)
    else:
        if len(mins) != n_bands:
            raise ValueError('Dataset minimum values must be specified for '
                             '"n_bands" (got %i values, needed %i)' %
                             (len(mins), n_bands))
        cfg['dataset']['min_values'] = np.asarray(mins)
    if isinstance(maxes, (float, int)):
        cfg['dataset']['max_values'] = np.asarray([maxes] * n_bands)
    else:
        if len(maxes) != n_bands:
            raise ValueError('Dataset maximum values must be specified for '
                             '"n_bands" (got %i values, needed %i)' %
                             (len(maxes), n_bands))
        cfg['dataset']['max_values'] = np.asarray(maxes)

    return cfg


def _parse_YATSM_config(cfg):
    """ Parse "YATSM:" configuration section
    """
    # Unpickle main predictor
    pred_method = cfg['YATSM']['prediction']
    cfg['YATSM']['estimator'] = {'prediction': pred_method}
    cfg['YATSM']['estimator']['object'] = _unpickle_predictor(
        _find_pickle(pred_method, cfg))
    # Grab estimator fit options
    cfg['YATSM']['estimator']['fit'] = cfg.get(
        pred_method, {}).get('fit', {}) or {}

    # Unpickle refit objects
    if cfg['YATSM'].get('refit', {}).get('prediction', None):
        # Restore pickles
        pickles = []
        fitopts = []
        for pred_method in cfg['YATSM']['refit']['prediction']:
            pickles.append(_unpickle_predictor(_find_pickle(pred_method, cfg)))
            fitopts.append(cfg.get(pred_method, {}).get('fit', {}) or {})
        cfg['YATSM']['refit']['prediction_object'] = pickles
        cfg['YATSM']['refit']['fit'] = fitopts
    # Fill in as empty refit
    else:
        refit = dict(prefix=[], prediction=[], prediction_object=[],
                     stay_regularized=[], fit=[])
        cfg['YATSM']['refit'] = refit

    # Check number of refits
    n_refit = len(cfg['YATSM']['refit']['prediction_object'])
    n_prefix = len(cfg['YATSM']['refit']['prefix'])
    if n_refit != n_prefix:
        raise KeyError('Must supply a prefix for all refix predictions '
                       '(%i vs %i)' % (n_refit, n_prefix))

    # Fill in "stay_regularized" -- default True
    reg = cfg['YATSM']['refit'].get('stay_regularized', None)
    if reg is None:
        cfg['YATSM']['refit']['stay_regularized'] = [True] * n_refit
    elif isinstance(reg, bool):
        cfg['YATSM']['refit']['stay_regularized'] = [reg] * n_refit

    return cfg


def _find_pickle(pickle, cfg):
    """ Return filename for pickle specified

    Pickle should either be from packaged estimators or specified as a section
    in the configuration file.
    """
    # Check if in packaged
    if pickle in packaged_regressions:
        pickle_path = find_packaged_regressor(pickle)
        logger.debug('Using pre-packaged prediction method "%s" from %s' %
                     (pickle, pickle_path))
        return pickle_path
    # Check if in configuration file
    elif pickle in cfg:
        if 'pickle' in cfg[pickle]:
            pickle_path = cfg[pickle]['pickle']
            logger.debug('Using prediction method "%s" from config file (%s)' %
                         (pickle, pickle_path))
            return pickle_path
        else:
            raise KeyError('Prediction method "%s" in config file, but no '
                           'path is given in "pickle" key' % pickle)
    else:
        raise KeyError('Prediction method "%s" is not a pre-packaged estimator'
                       ' nor is it specified as a section in config file' %
                       pickle)


def _unpickle_predictor(pickle):
    # Load sklearn objects
    reg = joblib.load(pickle)

    sklearn_attrs = ['fit', 'predict', 'get_params', 'set_params']
    if all([m in dir(reg) for m in sklearn_attrs]):
        return reg
    else:
        raise AttributeError('Cannot use prediction object from %s. Prediction'
                             ' objects must define the following attributes:\n'
                             '%s'
                             % (pickle, ', '.join(sklearn_attrs)))
