import logging
import os

import numpy as np
import six
import sklearn.externals.joblib as joblib
import yaml

from . import algorithms
from .regression.packaged import find_packaged_regressor, packaged_regressions

logger = logging.getLogger('yatsm')


def convert_config(cfg):
    """ Convert some configuration values to different values

    Args:
        cfg (dict): dict: dict of sub-dicts, each sub-dict containing
            configuration keys and values pertinent to a process or algorithm

    Returns:
        dict: configuration dict with some items converted to different objects

    Raises:
        KeyError: raise KeyError if configuration file is not specified
            correctly
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

    # Unpickle main predictor
    pred_method = cfg['YATSM']['prediction']
    if pred_method not in cfg:
        # Try to use pre-packaged regression method
        if pred_method not in packaged_regressions:
            raise KeyError(
                'Prediction method specified (%s) is not parameterized '
                'in configuration file nor available from the YATSM package'
                % pred_method)
        else:
            pred_method_path = find_packaged_regressor(pred_method)
            logger.debug('Using pre-packaged prediction method %s from %s' %
                         (pred_method, pred_method_path))
            cfg[pred_method] = {'pickle': pred_method_path}
    else:
        logger.debug('Predicting using "%s" pickle specified from '
                     'configuration file (%s)' %
                     (pred_method, cfg[pred_method]['pickle']))

    cfg['YATSM']['prediction_object'] = _unpickle_predictor(
        cfg[pred_method]['pickle'])

    # Unpickle refit objects
    if ('refit' in cfg['YATSM'] and
            cfg['YATSM']['refit'].get('prediction', None)):
        pickles = []
        for predictor in cfg['YATSM']['refit']['prediction']:
            if predictor in cfg:
                pickle_file = cfg[predictor]['pickle']
            elif predictor in packaged_regressions:
                pickle_file = find_packaged_regressor(predictor)
                logger.debug('Using pre-packaged prediction method %s from %s '
                             'for refitting' % (predictor, pickle_file))
            else:
                raise KeyError('Refit predictor specified (%s) is not a '
                               'pre-packaged predictor and is not specified '
                               'as section in config file' % predictor)
            pickles.append(_unpickle_predictor(pickle_file))
        cfg['YATSM']['refit']['prediction_object'] = pickles
    else:
        refit = dict(prefix=[], prediction=[], prediction_object=[])
        cfg['YATSM']['refit'] = refit

    return cfg


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
    if algo not in algorithms.available:
        raise NotImplementedError('Algorithm specified (%s) is not currently '
                                  'available' % algo)
    cfg['YATSM']['algorithm_cls'] = getattr(algorithms, algo)
    if not cfg['YATSM']['algorithm_cls']:
        raise KeyError('Could not find algorithm specified (%s) in '
                       '`yatsm.algorithms`' % algo)

    # Add in dummy phenology and classification dicts if not included
    if 'phenology' not in cfg:
        cfg['phenology'] = {'enable': False}

    if 'classification' not in cfg:
        cfg['classification'] = {'training_image': None}

    return convert_config(cfg)


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


def expand_envvars(d):
    """ Recursively convert lookup that look like environment vars in a dict

    This function things that environmental variables are values that begin
    with '$' and are evaluated with ``os.path.expandvars``. No exception will
    be raised if an environment variable is not set.

    Args:
        d (dict): expand environment variables used in the values of this
            dictionary

    Returns:
        dict: input dictionary with environment variables expanded

    """
    def check_envvar(k, v):
        """ Warn if value looks un-expanded """
        if '$' in v:
            logger.warning('Config key=value pair might still contain '
                            'environment variables: "%s=%s"' % (k, v))

    _d = d.copy()
    for k, v in six.iteritems(_d):
        if isinstance(v, dict):
            _d[k] = expand_envvars(v)
        elif isinstance(v, str):
            _d[k] = os.path.expandvars(v)
            check_envvar(k, v)
        elif isinstance(v, (list, tuple)):
            n_v = []
            for _v in v:
                if isinstance(_v, str):
                    _v = os.path.expandvars(_v)
                    check_envvar(k, _v)
                n_v.append(_v)
            _d[k] = n_v
    return _d
