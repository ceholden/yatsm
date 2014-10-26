#!/usr/bin/env python
""" Yet Another Time Series Model

Usage:
    yatsm.py [options] <location> <px> <py>

Options:
    --consecutive=<n>       Consecutive observations for change [default: 5]
    --threshold=<T>         Threshold for change [default: 2.56]
    --min_obs=<n>           Min number of obs per model [default: 1.5 * n_coef]
    --freq=<freq>           Sin/cosine frequencies [default: 1 2 3]
    --min_rmse=<rmse>       Minimum RMSE used in detection [default: None]
    --screening=<method>    Multi-temporal screening method [default: RLM]
    --lassocv               Use sklearn cross-validated LassoLarsIC
    --reverse               Run timeseries in reverse
    --test_indices=<bands>  Test bands [default: ALL]
    --omit_crit=<crit>      Critical value for omission test
    --omit_behavior=<b>     Omission test behavior [default: ALL]
    --omit_indices=<b>      Image indices used in omission test
    --plot_index=<b>        Index of band to plot for diagnostics
    --plot_ylim=<lim>       Plot y-limits
    -v --verbose            Show verbose debugging messages
    -h --help               Show help

Example:

    Display the results plotted with Band 5 for a pixel using 5 consecutive
        observations and 3 threshold for break detection. Each model's "trim"
        or minimum number of observations is 16 and we use one seasonal
        harmonic per year.

        > run_yatsm.py --consecutive=5 --threshold=3
        ...     --min_obs=16 --freq=1
        ...     --plot_index=5 --plot_ylim "1000 4000"
        ...     ../../landsat_stack/p022r049/images/ 125 125

"""
from __future__ import print_function, division

from datetime import datetime as dt
import logging
import math

from docopt import docopt

import numpy as np
import pandas as pd

from ggplot import *

from yatsm import YATSM, make_X
from ts_driver.timeseries_ccdc import CCDCTimeSeries

from IPython.core.debugger import Pdb


# Some constants
ndays = 365.25
fmask = 7

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def preprocess(location, px, py, freq):
    """ Read and preprocess Landsat data before analysis """
    # Load timeseries
    ts = CCDCTimeSeries(location, image_pattern='L*')
    ts.set_px(px)
    ts.set_py(py)
    ts.get_ts_pixel()

    # Get dates (datetime)
    x = ts.dates
    # Get data
    Y = ts.get_data(mask=False)

    # Filter out time series and remove Fmask
    clear = np.logical_and(Y[fmask, :] <= 1,
                           np.all(Y <= 10000, axis=0))
    Y = Y[:, clear][:fmask, :]
    x = x[clear]

    # Ordinal date
    ord_x = np.array(map(dt.toordinal, x))
    # Make X matrix
    X = make_X(ord_x, freq).T

    return (ts, X, Y, clear)


def make_plot():
    # Get qualitative color map
    import brewer2mpl
    # Color map ncolors goes from 3 - 9
    ncolors = min(9, max(3, len(yatsm.record)))
    # Repeat if number of segments > 9
    repeat = int(math.ceil(len(yatsm.record) / 9.0))
    colors = brewer2mpl.get_map('set1',
                                'qualitative',
                                ncolors).hex_colors * repeat
    if reverse:
        i_step = -1
    else:
        i_step = 1

    # Add in deleted obs
    deleted = ~np.in1d(Y[plot_index, :], yatsm.Y[plot_index, :])
    point_colors = np.repeat('black', X.shape[0])
    point_colors[deleted] = 'red'

    p = ggplot(aes('date', df.columns[plot_index]), df) + \
        geom_point(color=point_colors)

    for i, r in enumerate(yatsm.record):
        # Setup dummy dataframe for predictions
        ts_df = pd.DataFrame(
            {'x': np.arange(r['start'], r['end'], i_step)})
        # Get predictions
        ts_df['y'] = np.dot(r['coef'][:, plot_index],
                            make_X(ts_df['x'], freq))
        ts_df['date'] = np.array(
            [dt.fromordinal(int(_d)) for _d in ts_df['x']])
        # Add line to ggplot
        p = p + geom_line(aes('date', 'y'), ts_df, color=colors[i])
        # If there is a break in this timeseries, add it as vertical line
        if r['break'] != 0:
            p = p + \
                geom_vline(
                    xintercept=dt.fromordinal(r['break']), color='red')
    if plot_ylim:
        p = p + ylim(plot_ylim[0], plot_ylim[1])
    # Show graph
    if reverse:
        title = 'Reverse'
    else:
        title = 'Forward'
    print(p + ggtitle(title))


if __name__ == '__main__':
    args = docopt(__doc__)

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
    if min_rmse.lower() == 'none':
        min_rmse = None
    else:
        min_rmse = float(min_rmse)

    # Multi-temporal screening method
    screening = args['--screening']
    if screening not in YATSM.screening_types:
        raise TypeError('Unknown multi-temporal cloud screening type')

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

    # Debug level
    debug = args['--verbose']
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Get data and mask clouds
    ts, X, Y, clear = preprocess(location, px, py, freq)

    # Create dataframe
    df = pd.DataFrame(np.vstack((Y, X.T)).T,
                      columns=['b1', 'b2', 'b3', 'b4', 'b5', 'b7', 'b6'] +
                      ['B' + str(i) for i in range(X.shape[1])])
    df.index = df['B1']
    df['date'] = ts.dates[clear]

    if plot_index:
        p = ggplot(aes('date', df.columns[plot_index]), df) + geom_point()
        if plot_ylim:
            p = p + ylim(plot_ylim[0], plot_ylim[1])
        print(p)

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
                  screening=screening,
                  test_indices=test_indices,
                  lassocv=lassocv,
                  logger=logger)
    yatsm.run()

    breakpoints = yatsm.record['break'][yatsm.record['break'] != 0]

    print('Found {n} breakpoints'.format(n=breakpoints.size))
    if breakpoints.size > 0:
        print(breakpoints)

    if plot_index:
        make_plot()

    if omission_crit:
        print('Omission test (alpha = {a}):'.format(a=omission_crit))
        if isinstance(omission_bands, np.ndarray) or \
                isinstance(test_indices, np.ndarray) is not None:
            print('    {b} indices?:'.format(b=omission_bands if omission_bands
                                             else test_indices))
        print(yatsm.omission_test(crit=omission_crit,
                                  behavior=omission_behavior,
                                  indices=omission_bands))
