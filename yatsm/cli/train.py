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

from yatsm.cli import options
from yatsm.config_parser import parse_config_file
from yatsm import classifiers
from yatsm.classifiers import diagnostics
from yatsm import plots
from yatsm import reader
from yatsm import utils

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
    Train a classifier from `scikit-learn` on YATSM output and save result to
    file <model>. Dataset configuration is specified by <yatsm_config> and
    classifier and classifier parameters are specified by <classifier_config>.
    """
    if not model.endswith('.pkl'):
        model += '.pkl'
    if os.path.isfile(model) and not overwrite:
        logger.error('<model> exists and --overwrite was not specified')
        raise click.Abort()

    if seed:
        np.random.seed(seed)

    # Parse YATSM config
    dataset_config, yatsm_config = parse_config_file(config)

    if not dataset_config['training_image'] or \
            not os.path.isfile(dataset_config['training_image']):
        logger.error('Training data image {f} does not exist'.format(
            f=dataset_config['training_image']))
        raise click.Abort()

    # Parse classifier config
    algorithm_helper = classifiers.ini_to_algorthm(classifier_config)

    main(dataset_config, yatsm_config, algorithm_helper, model,
         diagnostics, n_fold, plot)


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
      row (np.ndarray): row pixel locations of `y`
      col (np.ndarray): column pixel locations of `y`
      labels (np.ndarraY): label of `y` if found, else None

    """
    # Find and parse training data
    roi = reader.read_image(dataset_config['training_image'])
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
            dataset_config['training_start'],
            dataset_config['training_date_format']).toordinal()
        training_end = dt.strptime(
            dataset_config['training_end'],
            dataset_config['training_date_format']).toordinal()
    except:
        logger.error('Failed to parse training data start or end dates')
        raise

    # Loop through samples in ROI extracting features
    mask = ~np.in1d(roi, dataset_config['roi_mask_values']).reshape(roi.shape)
    row, col = np.where(mask)
    y = roi[row, col]

    X = []
    out_y = []
    out_row = []
    out_col = []

    rec = None
    _row_previous = None
    for _row, _col, _y in izip(row, col, y):
        # Load result
        if _row != _row_previous:
            try:
                rec = np.load(utils.get_output_name(
                    dataset_config, _row))['record']
                _row_previous = _row
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
                label {l} at x/y {x}/{y}'.format(l=_y, x=_col, y=_row))

        # Extract coefficients with intercept term rescaled
        coef = rec[i]['coef'][0, :]
        coef[0, :] = (coef[0, :] +
                      coef[1, :] * (rec[i]['start'] + rec[i]['end']) / 2.0)

        X.append(np.concatenate(
            (coef.reshape(coef.size), rec[i]['rmse'][0])))
        out_y.append(_y)
        out_row.append(_row)
        out_col.append(_col)

    if not out_y:
        logger.error('Could not find any matching timeseries segments')
        raise click.Abort()
    logger.info('Found matching time segments for {m} out of {n} labels'.
                format(m=len(out_y), n=y.size))

    out_row = np.array(out_row)
    out_col = np.array(out_col)

    if labels is not None:
        labels = labels[out_row, out_col]

    return (np.array(X), np.array(out_y),
            out_row, out_col, labels)


def algo_diagnostics(dataset_config, yatsm_config, X, y,
                     row, col, algo, n_fold, make_plots=True):
    """ Display algorithm diagnostics for a given X and y

    Args:
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
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
        logger.info('Out of Bag score: {p}'.format(p=algo.oob_score_))

    kfold_summary = np.zeros((0, 2))

    logger.info('<----------------------->')
    logger.info('KFold crossvalidation scores:')
    kf = KFold(y.size, n_folds=n_fold)
    kfold_summary = np.vstack((kfold_summary,
                              diagnostics.kfold_scores(X, y, algo, kf)
                               ))

    logger.info('<----------------------->')
    logger.info('Stratified KFold crossvalidation scores:')
    kf = StratifiedKFold(y, n_folds=n_fold)
    kfold_summary = np.vstack((kfold_summary,
                              diagnostics.kfold_scores(X, y, algo, kf)
                               ))

    logger.info('<----------------------->')
    logger.info('Spatialized shuffled KFold crossvalidation scores:')
    kf = diagnostics.SpatialKFold(y, row, col, n_folds=n_fold, shuffle=True)
    kfold_summary = np.vstack((kfold_summary,
                              diagnostics.kfold_scores(X, y, algo, kf)
                               ))

    if make_plots:
        test_names = ['KFold',
                      'Stratified KFold',
                      'Spatial KFold (shuffle)'
                      ]
        plots.plot_crossvalidation_scores(kfold_summary, test_names)

    logger.info('<----------------------->')
    if hasattr(algo, 'feature_importances_'):
        logger.info('Feature importance:')
        logger.info(algo.feature_importances_)
        if make_plots:
            plots.plot_feature_importance(algo, dataset_config, yatsm_config)


def main(dataset_config, yatsm_config, algo, model_filename,
         run_diagnostics, n_fold, make_plots=True):
    """ YATSM trainining main

    Args:
      dataset_config (dict): options for the dataset
      yatsm_config (dict): options for the change detection algorithm
      algo (sklearn classifier): classification algorithm helper class
      model_filename (str): filename for pickled algorithm object
      run_diagnostics (bool): Run KFold diagnostics
      n_fold (int): number of folds for crossvalidation
      make_plots (bool, optional): show diagnostic plots (default: True)

    """
    # Cache file for training data
    has_cache = False
    if dataset_config['cache_training']:
        # If doesn't exist, retrieve it
        if not os.path.isfile(dataset_config['cache_training']):
            logger.info('Could not retrieve cache file for Xy')
            logger.info('    file: {f}'.format(
                        f=dataset_config['cache_training']))
        else:
            logger.info('Restoring X/y from cache file')
            has_cache = True

    # Check if we need to regenerate the cache file because training data is
    #   newer than the cache
    regenerate_cache = is_cache_old(dataset_config['cache_training'],
                                    dataset_config['training_image'])
    if regenerate_cache:
        logger.warning('Existing cache file older than training data ROI')
        logger.warning('Regenerating cache file')

    if not has_cache or regenerate_cache:
        logger.debug('Reading in X/y')
        X, y, row, col, labels = get_training_inputs(dataset_config)
        logger.debug('Done reading in X/y')
    else:
        logger.debug('Reading in X/y from cache file {f}'.format(
            f=dataset_config['cache_training']))
        with np.load(dataset_config['cache_training']) as f:
            X = f['X']
            y = f['y']
            row = f['row']
            col = f['col']
            labels = f['labels']
        logger.debug('Read in X/y from cache file {f}'.format(
            f=dataset_config['cache_training']))

    # If cache didn't exist but is specified, create it for first time
    if not has_cache and dataset_config['cache_training']:
        logger.info('Saving X/y to cache file {f}'.format(
            f=dataset_config['cache_training']))
        try:
            np.savez(dataset_config['cache_training'],
                     X=X, y=y, row=row, col=col, labels=labels)
        except:
            logger.error('Could not save X/y to cache file')
            raise

    # Do modeling
    logger.info('Training classifier')
    algo.fit(X, y)

    # Serialize algorithm to file
    logger.info('Pickling classifier with sklearn.externals.joblib')
    joblib.dump(algo, model_filename, compress=3)

    # Diagnostics
    if run_diagnostics:
        algo_diagnostics(X, y, row, col, algo, n_fold, make_plots)
