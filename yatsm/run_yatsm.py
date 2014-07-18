#!/usr/bin/env python
""" Yet Another Time Series Model

Usage:
    yatsm.py [options] <location> <px> <py>

Options:
    --consecutive=<n>       Consecutive observations to find change [default: 5]
    --threshold=<T>         Threshold for change [default: 2.56]
    --min_obs=<n>           Min number of obs per model [default: 1.5 * n_coef]
    --freq=<freq>           Sin/cosine frequencies [default: 1 2 3]
    --lassocv               Use sklearn cross-validated LassoCV
    --reverse               Run timeseries in reverse
    --plot_band=<b>         Band to plot for diagnostics [default: None]
    --plot_ylim=<lim>       Plot y-limits [default: None]
    --ensemble=<n>          Number of change runs [default: 1]
    --threshvar=<n>         Threshold variance [default: 0.5]
    --consecvar=<n>         Consecutive observation variance [default: 1]
    --ensemble_order        Run ensemble forward and reverse
    --debug                 Show verbose debugging messages
    -h --help               Show help

Example:


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

# Some constants
ndays = 365.25
fmask = 7


def preprocess(location, px, py):
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
    clear = Y[7, :] <= 1
    Y = Y[:, clear][:fmask, :]
    x = x[clear]

    # Ordinal date
    ord_x = np.array(map(dt.toordinal, x))
    # Make X matrix
    X = make_X(ord_x, freq).T

    return (ts, X, Y, clear)

if __name__ == '__main__':
    args = docopt(__doc__)

    location = args['<location>']
    px = int(args['<px>'])
    py = int(args['<py>'])

    print('Working on: px={px}, py={py}'.format(px=px, py=py))

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

    # Cross-validated Lasso
    lassocv = args['--lassocv']
    # Reverse run?
    reverse = args['--reverse']

    # Plot band for debug
    plot_band = args['--plot_band']
    if plot_band == 'None':
        plot_band = None
    else:
        plot_band = int(plot_band)

    plot_ylim = args['--plot_ylim']
    if plot_ylim == 'None':
        plot_ylim = None
    else:
        plot_ylim = [int(n) for n in
                     plot_ylim.replace(' ', ',').split(',') if n != '']

    # Ensemble runs
    ensemble = args['--ensemble']
    if ensemble == '1':
        ensemble = None
    else:
        ensemble = int(ensemble)
    ensemble_order = args['--ensemble_order']

    # Debug level
    debug = args['--debug']
    if debug:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.WARNING

    # Get data and mask clouds
    ts, X, Y, clear = preprocess(location, px, py)

    # Create dataframe
    df = pd.DataFrame(np.vstack((Y, X.T)).T,
                      columns=['b1', 'b2', 'b3', 'b4', 'b5', 'b7', 'b6'] +
                      ['B' + str(i) for i in range(X.shape[1])])
    df.index = df['B1']
    df['date'] = ts.dates[clear]

    if plot_band:
        p = ggplot(aes('date', 'b' + str(plot_band)), df) + geom_point()
        if plot_ylim:
            p = p + ylim(plot_ylim[0], plot_ylim[1])
        print(p)

    if reverse:
        yatsm = YATSM(np.flipud(X), np.fliplr(Y),
                      consecutive=consecutive,
                      threshold=threshold,
                      min_obs=min_obs,
                      lassocv=lassocv,
                      loglevel=loglevel)
    else:
        yatsm = YATSM(X, Y,
                      consecutive=consecutive,
                      threshold=threshold,
                      min_obs=min_obs,
                      lassocv=lassocv,
                      loglevel=loglevel)
    yatsm.run()

    breakpoints = yatsm.record['break']

    print('Found {n} breakpoints'.format(n=breakpoints.size))
    print(breakpoints)

    if plot_band:
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

        p = ggplot(aes('date', 'b' + str(plot_band)), df) + \
            geom_point()

        for i, r in enumerate(yatsm.record):
            # Setup dummy dataframe for predictions
            ts_df = pd.DataFrame(
                {'x': np.arange(r['start'], r['end'], i_step)})
            # Get predictions
            ts_df['y'] = np.dot(r['coef'][:, plot_band - 1],
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
