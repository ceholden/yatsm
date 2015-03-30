#!/usr/bin/env python
""" Yet Another Time Series Model

Usage:
    run_yatsm.py [options] <location> <px> <py>

Algorithm options:
    --consecutive=<n>       Consecutive observations for change [default: 5]
    --threshold=<T>         Threshold for change [default: 2.56]
    --min_obs=<n>           Min number of obs per model [default: 1.5 * n_coef]
    --freq=<freq>           Sin/cosine frequencies [default: 1 2 3]
    --min_rmse=<rmse>       Minimum RMSE used in detection
    --retrain_time=<n>      Wait <n> days to update model [default: 365.25]
    --screening=<method>    Multi-temporal screening method [default: RLM]
    --screening_crit=<t>    Screening critical value [default: 400.0]
    --remove_noise          Remove noise during monitoring
    --dynamic_rmse          Vary RMSE as a function of day of year
    --lassocv               Use sklearn cross-validated LassoLarsIC
    --reverse               Run timeseries in reverse
    --test_indices=<bands>  Test indices [default: ALL]
    --commit_alpha=<alpha>  Alpha for F_crit in Chow commission test
    --omit_crit=<crit>      Critical value for omission test
    --omit_behavior=<b>     Omission test behavior [default: ALL]
    --omit_indices=<b>      Image indices used in omission test
    --pheno                 Predict phenology metrics using default parameters

Plotting options:
    --plot_index=<b>        Index of band to plot for diagnostics
    --plot_ylim=<lim>       Plot y-limits
    --plot_style=<style>    Plot style [default: ggplot]

Generic options:
    --image_pattern=<p>     Stack image filename pattern [default: L*stack]
    -v --verbose            Show verbose debugging messages
    --version               Print program version and exit
    -h --help               Show help

Example:

    Display the results plotted with Band 5 for a pixel using 5 consecutive
        observations and 3 threshold for break detection. Each model's "trim"
        or minimum number of observations is 16 and we use two seasonal
        harmonics per year. The plot uses XKCD styling!

    > run_yatsm.py --consecutive=5 --threshold=3 --min_obs=16
    ... --freq="1, 2" --min_rmse 150 --test_indices "2 4 5" --screening RLM
    ... --plot_index=4 --plot_style xkcd
    ... ../landsat_stack/p022r049/images/ 150 50

"""
from __future__ import print_function, division

from datetime import datetime as dt
import logging
import math
import os
import sys

from docopt import docopt

import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Handle running as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.utils import make_X
from yatsm.reader import find_stack_images, read_pixel_timeseries
from yatsm.yatsm import YATSM

import sklearn.linear_model

# Some constants
ndays = 365.25
fmask = 7

plot_styles = plt.style.available + [u'xkcd']

# Set default size to 11" x 6.798 (golden ratio)
plt.rcParams['figure.figsize'] = 11, 6.798

FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')


def preprocess(location, px, py, freq,
               image_pattern='L*stack'):
    """ Read and preprocess Landsat data before analysis """
    dates, files = find_stack_images(location,
                                     image_pattern=image_pattern)

    x = np.array(dates)
    Y = read_pixel_timeseries(files, px, py)

    # Filter out time series and remove Fmask
    clear = np.logical_and(Y[fmask, :] <= 1,
                           np.all(Y <= 10000, axis=0))
    Y = Y[:, clear][:fmask, :]
    x = x[clear]

    # Ordinal date
    ord_x = np.array(map(dt.toordinal, x))
    # Make X matrix
    X = make_X(ord_x, freq).T

    return X, Y, clear


def plot_dataset():
    """ Plots just the dataset before fitting """
    plt.plot(dates, Y[plot_index, :], 'ko')
    plt.ylim(plot_ylim)
    plt.xlabel('Time')
    plt.ylabel('Band {i}'.format(i=plot_index + 1))


def plot_results():
    # Add in deleted obs
    deleted = np.in1d(X[:, 1], yatsm.X[:, 1], invert=True)

    plot_dataset()
    plt.plot(dates[deleted], Y[plot_index, deleted], 'ro')

    # Get qualitative color map for model segment lines
    # Color map ncolors goes from 3 - 9
    ncolors = min(9, max(3, len(yatsm.record)))
    # Repeat if number of segments > 9
    repeat = int(math.ceil(len(yatsm.record) / 9.0))
    fit_colors = brewer2mpl.get_map('set1',
                                    'qualitative',
                                    ncolors).hex_colors * repeat

    # Direction for prediction x
    step = -1 if reverse else 1

    for i, r in enumerate(yatsm.record):
        # Predict
        mx = np.arange(r['start'], r['end'], step)
        my = np.dot(r['coef'][:, plot_index],
                    make_X(mx, freq))
        mx_date = np.array([dt.fromordinal(int(_x)) for _x in mx])

        plt.plot(mx_date, my, fit_colors[i])

        idx = np.where((yatsm.X[:, 1] >= r['start']) & (yatsm.X[:, 1] <= r['end']))[0]
        sklearn_lasso = sklearn.linear_model.Lasso(alpha=20).fit(yatsm.X[idx, :], yatsm.Y[plot_index, idx])

        plt.plot(mx_date, sklearn_lasso.predict(make_X(mx, freq).T), fit_colors[i], ls='dashed', lw=3)

        # from IPython.core.debugger import Pdb
        # Pdb().set_trace()

        if r['break'] != 0:
            break_date = dt.fromordinal(int(r['break']))
            break_i = np.where(X[:, 1] == r['break'])[0]
            if not plot_ylim:
                _plot_ylim = (Y[plot_index, :].min(), Y[plot_index, :].max())
            else:
                _plot_ylim = plot_ylim

            plt.vlines(break_date, _plot_ylim[0], _plot_ylim[1], 'r')
            plt.plot(break_date, Y[plot_index, break_i],
                     'ro', mec='r', mfc='none', ms=10, mew=5)


def plot_phenology():
    if not plot_ylim:
        _plot_ylim = (Y[plot_index, :].min(), Y[plot_index, :].max())
    else:
        _plot_ylim = plot_ylim

    # Break up into year/doy
    yeardoy = pheno.ordinal2yeardoy(yatsm.X[:, 1].astype(np.uint32))

    # Plot predicted pheno
    repeat = int(math.ceil(len(yatsm.record) / 9.0))
    fit_colors = brewer2mpl.get_map(
        'set1', 'qualitative', 9).hex_colors * repeat

    for i, r in enumerate(yatsm.record):
        # Plot data within record
        index = np.where((yatsm.X[:, 1] >= r['start']) &
                         (yatsm.X[:, 1] <= r['end']))
        plt.scatter(yeardoy[index, 1], yatsm.Y[plot_index, index],
                    c=fit_colors[i])

        # Plot spline'd EVI scaled to min/max of plot
        scaled_evi = (r['spline_evi'] * (_plot_ylim[1] - _plot_ylim[0]))
        plt.plot(np.arange(1, 366), scaled_evi, color=fit_colors[i],
                 ls='-')

        plt.vlines(r['spring_doy'], _plot_ylim[0], _plot_ylim[1],
                   fit_colors[i], linestyles='dashed', lw=3)
        plt.vlines(r['autumn_doy'], _plot_ylim[0], _plot_ylim[1],
                   fit_colors[i], linestyles='dashdot', lw=3)

        print('Segment {i}:')
        print('    spring: {d}'.format(i=i, d=r['spring_doy']))
        print('    autumn: {d}'.format(i=i, d=r['autumn_doy']))
        print('    correlation: {d}'.format(i=i, d=r['pheno_cor']))
        print('    # observatiosn: {n}'.format(n=r['pheno_nobs']))

    plt.xlim(0, 366)
    plt.ylim(plot_ylim)
    plt.ylabel('Band {i}'.format(i=plot_index + 1))
    plt.xlabel('Day of Year')


if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    location = args['<location>']
    px = int(args['<px>'])
    py = int(args['<py>'])

    logger.info('Working on: px={px}, py={py}'.format(px=px, py=py))

    # Consecutive observations
    consecutive = int(args['--consecutive'])

    # Threshold for change
    threshold = float(args['--threshold'])

    # Minimum number of observations per segment
    min_obs = args['--min_obs']
    if min_obs == '1.5 * n_coef':
        min_obs = None
    else:
        min_obs = int(args['--min_obs'])

    # Sin/cosine frequency for independent variables
    freq = args['--freq']
    freq = [int(n) for n in freq.replace(' ', ',').split(',') if n != '']

    # Minimum RMSE
    min_rmse = args['--min_rmse']
    if min_rmse:
        min_rmse = float(min_rmse)

    # Retraining time
    retrain_time = float(args['--retrain_time'])

    # Multi-temporal screening method
    screening = args['--screening']
    if screening not in YATSM.screening_types:
        raise TypeError('Unknown multi-temporal cloud screening type')
    screening_crit = float(args['--screening_crit'])
    remove_noise = args['--remove_noise']

    dynamic_rmse = args['--dynamic_rmse']

    # Cross-validated Lasso
    lassocv = args['--lassocv']
    # Reverse run?
    reverse = args['--reverse']

    # Test bands
    test_indices = args['--test_indices']
    if test_indices.lower() == 'all':
        test_indices = None
    else:
        test_indices = np.array([int(b) for b in
                                test_indices.replace(' ', ',').split(',')
                                if b != ''])

    # Omission test
    omission_crit = args['--omit_crit']
    if omission_crit:
        omission_crit = float(omission_crit)
        if omission_crit >= 1 or omission_crit <= 0:
            raise ValueError(
                'Omission test critical value must be between 0 - 1')

    omission_behavior = args['--omit_behavior']
    if omission_behavior.lower() not in ['all', 'any']:
        raise TypeError('Unknown omission behavior type (must be ANY or ALL)')

    omission_bands = args['--omit_indices']
    if omission_bands:
        omission_bands = [int(b) for b in
                          omission_bands.replace(' ', ',').split(',')
                          if b != '']

    # Plot band for debug
    plot_index = args['--plot_index']
    if plot_index:
        plot_index = int(plot_index)

    plot_ylim = args['--plot_ylim']
    if plot_ylim:
        plot_ylim = [int(n) for n in
                     plot_ylim.replace(' ', ',').split(',') if n != '']

    plot_style = args['--plot_style']
    if plot_style not in plot_styles:
        raise ValueError('Unknown style. Available styles are {s}'.format(
            s=plot_styles))
    if plot_style == 'xkcd':
        style_context = plt.xkcd()
    else:
        style_context = plt.style.context(plot_style)

    # Stack image filename pattern
    image_pattern = args['--image_pattern']

    # Debug level
    debug = args['--verbose']
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

### BEGIN ACTUAL WORK
    # Get data and mask clouds
    X, Y, clear = preprocess(location, px, py, freq,
                             image_pattern=image_pattern)
    dates = np.array([dt.fromordinal(int(_x)) for _x in X[:, 1]])

    if isinstance(plot_index, int):
        with style_context:
            plot_dataset()
            plt.title('Timeseries')
            plt.tight_layout()
            plt.show()

    # Run model
    if reverse:
        _X = np.flipud(X)
        _Y = np.fliplr(Y)
    else:
        _X = X
        _Y = Y

    yatsm = YATSM(_X, _Y,
                  consecutive=consecutive,
                  threshold=threshold,
                  min_obs=min_obs,
                  min_rmse=min_rmse,
                  test_indices=test_indices,
                  retrain_time=retrain_time,
                  screening=screening,
                  screening_crit=screening_crit,
                  remove_noise=remove_noise,
                  dynamic_rmse=dynamic_rmse,
                  lassocv=lassocv,
                  logger=logger)
    yatsm.run()

    breakpoints = yatsm.record['break'][yatsm.record['break'] != 0]

    print('Found {n} breakpoints'.format(n=breakpoints.size))
    if breakpoints.size > 0:
        for i, bp in enumerate(breakpoints):
            print('Break {0}: {1}'.format(
                i, dt.fromordinal(bp).strftime('%Y-%m-%d')))

    # Renew the generator for style
    if plot_style == 'xkcd':
        style_context = plt.xkcd()
    else:
        style_context = plt.style.context(plot_style)

    if isinstance(plot_index, int):
        with style_context:
            plot_results()
            plt.tight_layout()
            plt.title('Modeled Timeseries')
            plt.show()

    if args['--commit_alpha']:
        yatsm.record = yatsm.commission_test(float(args['--commit_alpha']))

        # Renew the generator for style
        if plot_style == 'xkcd':
            style_context = plt.xkcd()
        else:
            style_context = plt.style.context(plot_style)

        if isinstance(plot_index, int):
            with style_context:
                plot_results()
                plt.tight_layout()
                plt.title('Modeled Timeseries (with commission test)')
                plt.show()

    if args['--pheno']:
        import yatsm.phenology as pheno
        yatsm.record = pheno.LongTermMeanPhenology(yatsm).fit()

        # Renew the generator for style
        if plot_style == 'xkcd':
            style_context = plt.xkcd()
        else:
            style_context = plt.style.context(plot_style)
        with style_context:
            plot_phenology()
            plt.tight_layout()
            plt.title('Modeled Phenology')
            plt.show()

    if omission_crit:
        print('Omission test (alpha = {a}):'.format(a=omission_crit))
        if isinstance(omission_bands, np.ndarray) or \
                isinstance(test_indices, np.ndarray) is not None:
            print('    {b} indices?:'.format(b=omission_bands if omission_bands
                                             else test_indices))
        print(yatsm.omission_test(crit=omission_crit,
                                  behavior=omission_behavior,
                                  indices=omission_bands))
