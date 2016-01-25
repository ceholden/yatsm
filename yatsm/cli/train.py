""" Command line interface for training classifiers on YATSM output """
from datetime import datetime as dt
from itertools import izip
import logging
import os

import click
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal

from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.externals import joblib

from . import options
from ..config_parser import parse_config_file
from ..classifiers import cfg_to_algorithm, diagnostics
from ..errors import TrainingDataException
from .. import io, plots, utils

logger = logging.getLogger('yatsm')

gdal.AllRegister()
gdal.UseExceptions()

if hasattr(plt, 'style') and 'ggplot' in plt.style.available:
    plt.style.use('ggplot')


@click.command(short_help='Train classifier on YATSM output')
@options.arg_config_file
@click.argument('classifier_config', metavar='<classifier_config>', nargs=1,
                type=click.Path(exists=True, readable=True,
                                dir_okay=False, resolve_path=True))
@click.argument('model', metavar='<model>', nargs=1,
                type=click.Path(writable=True, dir_okay=False,
                                resolve_path=True))
@click.option('--kfold', 'n_fold', nargs=1, type=click.INT, default=3,
              help='Number of folds in cross validation (default: 3)')
@click.option('--seed', nargs=1, type=click.INT,
              help='Random number generator seed')
@click.option('--plot', is_flag=True, help='Show diagnostic plots')
@click.option('--diagnostics', is_flag=True, help='Run K-Fold diagnostics')
@click.option('--overwrite', is_flag=True, help='Overwrite output model file')
@click.pass_context
def train(ctx, config, classifier_config, model, n_fold, seed,
          plot, diagnostics, overwrite):
    """
    Train a classifier from ``scikit-learn`` on YATSM output and save result to
    file <model>. Dataset configuration is specified by <yatsm_config> and
    classifier and classifier parameters are specified by <classifier_config>.
    """
    # Setup
    if not model.endswith('.pkl'):
        model += '.pkl'
    if os.path.isfile(model) and not overwrite:
        raise click.ClickException('<model> exists and --overwrite was not '
                                   'specified')

    if seed:
        np.random.seed(seed)

    # Parse config & algorithm config
    cfg = parse_config_file(config)
    algo, algo_cfg = cfg_to_algorithm(classifier_config)

    training_image = cfg['classification']['training_image']
    if not training_image or not os.path.isfile(training_image):
        raise click.ClickException('Training data image {} does not exist'
                                   .format(training_image))

    # Find information from results -- e.g., design info
    attrs = find_result_attributes(cfg)
    cfg['YATSM'].update(attrs)

    # Cache file for training data
    has_cache = False
    training_cache = cfg['classification']['cache_training']
    if training_cache:
        # If doesn't exist, retrieve it
        if not os.path.isfile(training_cache):
            logger.info('Could not retrieve cache file for Xy')
            logger.info('    file: %s' % training_cache)
        else:
            logger.info('Restoring X/y from cache file')
            has_cache = True

    training_image = cfg['classification']['training_image']
    # Check if we need to regenerate the cache file because training data is
    #   newer than the cache
    regenerate_cache = is_cache_old(training_cache, training_image)
    if regenerate_cache:
        logger.warning('Existing cache file older than training data ROI')
        logger.warning('Regenerating cache file')

    if not has_cache or regenerate_cache:
        logger.debug('Reading in X/y')
        X, y, row, col, labels = get_training_inputs(cfg)
        logger.debug('Done reading in X/y')
    else:
        logger.debug('Reading in X/y from cache file %s' % training_cache)
        with np.load(training_cache) as f:
            X = f['X']
            y = f['y']
            row = f['row']
            col = f['col']
            labels = f['labels']
        logger.debug('Read in X/y from cache file %s' % training_cache)

    # If cache didn't exist but is specified, create it for first time
    if not has_cache and training_cache:
        logger.info('Saving X/y to cache file %s' % training_cache)
        try:
            np.savez(training_cache,
                     X=X, y=y, row=row, col=col, labels=labels)
        except Exception as e:
            raise click.ClickException('Could not save X/y to cache file ({})'
                                       .format(e))

    # Do modeling
    logger.info('Training classifier')
    algo.fit(X, y, **algo_cfg.get('fit', {}))

    # Serialize algorithm to file
    logger.info('Pickling classifier with sklearn.externals.joblib')
    joblib.dump(algo, model, compress=3)

    # Diagnostics
    if diagnostics:
        algo_diagnostics(cfg, X, y, row, col, algo, n_fold, plot)


def is_cache_old(cache_file, training_file):
    """ Indicates if cache file is older than training data file

    Args:
        cache_file (str): filename of the cache file
        training_file (str): filename of the training data file

    Returns:
        bool: True if the cache file is older than the training data file
            and needs to be updated; False otherwise

    """
    if cache_file and os.path.isfile(cache_file):
        return os.stat(cache_file).st_mtime < os.stat(training_file).st_mtime
    else:
        return False


def find_result_attributes(cfg):
    """ Return result attributes relevant for training a classifier

    At this time, the only relevant information is the design information,
    ``design (OrderedDict)`` and ``design_matrix (str)``

    Args:
        cfg (dict): YATSM configuration dictionary

    Returns:
        dict: dictionary of result attributes

    """
    attrs = {
        'design': None,
        'design_matrix': None
    }

    for result in utils.find_results(cfg['dataset']['output'],
                                     cfg['dataset']['output_prefix'] + '*'):
        try:
            md = np.load(result)['metadata'].item()
            attrs['design'] = md['YATSM']['design']
            attrs['design_matrix'] = md['YATSM']['design_matrix']
        except:
            pass
        else:
            return attrs
    raise AttributeError('Could not find following attributes in results: {}'
                         .format(attrs.keys()))


def get_training_inputs(cfg, exit_on_missing=False):
    """ Returns X features and y labels specified in config file

    Args:
        cfg (dict): YATSM configuration dictionary
        exit_on_missing (bool, optional): exit if input feature cannot be found

    Returns:
        X (np.ndarray): matrix of feature inputs for each training data sample
        y (np.ndarray): array of labeled training data samples
        row (np.ndarray): row pixel locations of `y`
        col (np.ndarray): column pixel locations of `y`
        labels (np.ndarraY): label of `y` if found, else None

    """
    # Find and parse training data
    roi = io.read_image(cfg['classification']['training_image'])
    logger.debug('Read in training data')
    if len(roi) == 2:
        logger.info('Found labels for ROIs -- including in output')
        labels = roi[1]
    else:
        roi = roi[0]
        labels = None

    # Determine start and end dates of training sample relevance
    try:
        training_start = dt.strptime(
            cfg['classification']['training_start'],
            cfg['classification']['training_date_format']).toordinal()
        training_end = dt.strptime(
            cfg['classification']['training_end'],
            cfg['classification']['training_date_format']).toordinal()
    except:
        logger.error('Failed to parse training data start or end dates')
        raise

    # Loop through samples in ROI extracting features
    mask_values = cfg['classification']['roi_mask_values']
    mask = ~np.in1d(roi, mask_values).reshape(roi.shape)
    row, col = np.where(mask)
    y = roi[row, col]

    X = []
    out_y = []
    out_row = []
    out_col = []

    _row_previous = None
    for _row, _col, _y in izip(row, col, y):
        # Load result
        if _row != _row_previous:
            output_name = utils.get_output_name(cfg['dataset'], _row)
            try:
                rec = np.load(output_name)['record']
                _row_previous = _row
            except:
                logger.error('Could not open saved result file %s' %
                             output_name)
                if exit_on_missing:
                    raise
                else:
                    continue

        # Find intersecting time segment
        i = np.where((rec['start'] < training_start) &
                     (rec['end'] > training_end) &
                     (rec['px'] == _col))[0]

        if i.size == 0:
            logger.debug('Could not find model for label %i at x/y %i/%i' %
                         (_y, _col, _row))
            continue
        elif i.size > 1:
            raise TrainingDataException(
                'Found more than one valid model for label %i at x/y %i/%i' %
                (_y, _col, _row))

        # Extract coefficients with intercept term rescaled
        coef = rec[i]['coef'][0, :]
        coef[0, :] = (coef[0, :] +
                      coef[1, :] * (rec[i]['start'] + rec[i]['end']) / 2.0)

        X.append(np.concatenate((coef.reshape(coef.size), rec[i]['rmse'][0])))
        out_y.append(_y)
        out_row.append(_row)
        out_col.append(_col)

    out_row = np.array(out_row)
    out_col = np.array(out_col)

    if labels is not None:
        labels = labels[out_row, out_col]

    return np.array(X), np.array(out_y), out_row, out_col, labels


def algo_diagnostics(cfg, X, y,
                     row, col, algo, n_fold, make_plots=True):
    """ Display algorithm diagnostics for a given X and y

    Args:
        cfg (dict): YATSM configuration dictionary
        X (np.ndarray): X feature input used in classification
        y (np.ndarray): y labeled examples
        row (np.ndarray): row pixel locations of `y`
        col (np.ndarray): column pixel locations of `y`
        algo (sklearn classifier): classifier used from scikit-learn
        n_fold (int): number of folds for crossvalidation
        make_plots (bool, optional): show diagnostic plots (default: True)

    """
    # Print algorithm diagnostics without crossvalidation
    logger.info('<----- DIAGNOSTICS ----->')
    if hasattr(algo, 'oob_score_'):
        logger.info('Out of Bag score: %f' % algo.oob_score_)

    kfold_summary = np.zeros((0, 2))
    test_names = ['KFold', 'Stratified KFold', 'Spatial KFold (shuffle)']

    def report(kf):
        logger.info('<----------------------->')
        logger.info('%s crossvalidation scores:' % kf.__class__.__name__)
        try:
            scores = diagnostics.kfold_scores(X, y, algo, kf)
        except Exception as e:
            logger.warning('Could not perform %s cross-validation: %s' %
                           (kf.__class__.__name__, e))
            return (np.nan, np.nan)
        else:
            return scores

    kf = KFold(y.size, n_folds=n_fold)
    kfold_summary = np.vstack((kfold_summary, report(kf)))

    kf = StratifiedKFold(y, n_folds=n_fold)
    kfold_summary = np.vstack((kfold_summary, report(kf)))

    kf = diagnostics.SpatialKFold(y, row, col, n_folds=n_fold, shuffle=True)
    kfold_summary = np.vstack((kfold_summary, report(kf)))

    if make_plots:
        plots.plot_crossvalidation_scores(kfold_summary, test_names)

    logger.info('<----------------------->')
    if hasattr(algo, 'feature_importances_'):
        logger.info('Feature importance:')
        logger.info(algo.feature_importances_)
        if make_plots:
            plots.plot_feature_importance(algo, cfg)
