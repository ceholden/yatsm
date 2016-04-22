""" Command line interface for running YATSM algorithms on individual pixels
"""
import datetime as dt
import logging
import re

import click
import matplotlib as mpl
import matplotlib.cm  # noqa
import matplotlib.pyplot as plt
import numpy as np
import patsy
import yaml

from . import options, console
from ..algorithms import postprocess
from ..config_parser import convert_config, parse_config_file
from ..io import read_pixel_timeseries
from ..utils import csvfile_to_dataframe, get_image_IDs
from ..regression.transforms import harm  # noqa

avail_plots = ['TS', 'DOY', 'VAL']

_DEFAULT_PLOT_CMAP = 'viridis'
PLOT_CMAP = _DEFAULT_PLOT_CMAP
if PLOT_CMAP not in mpl.cm.cmap_d:
    PLOT_CMAP = 'cubehelix'


plot_styles = []
if hasattr(mpl, 'style'):
    plot_styles = mpl.style.available
if hasattr(plt, 'xkcd'):
    plot_styles.append('xkcd')

logger = logging.getLogger('yatsm')


@click.command(short_help='Run YATSM algorithm on individual pixels')
@options.arg_config_file
@click.argument('px', metavar='<px>', nargs=1, type=click.INT)
@click.argument('py', metavar='<py>', nargs=1, type=click.INT)
@click.option('--band', metavar='<n>', nargs=1, type=click.INT, default=1,
              show_default=True, help='Band to plot')
@click.option('--plot', default=('TS',), multiple=True, show_default=True,
              type=click.Choice(avail_plots), help='Plot type')
@click.option('--ylim', metavar='<min> <max>', nargs=2, type=float,
              show_default=True, help='Y-axis limits')
@click.option('--style', metavar='<style>', default='ggplot',
              show_default=True, type=click.Choice(plot_styles),
              help='Plot style')
@click.option('--cmap', metavar='<cmap>', default=PLOT_CMAP,
              show_default=True, help='DOY/VAL plot colormap')
@click.option('--embed', is_flag=True,
              help='Drop to (I)Python interpreter at various points')
@click.option('--seed', help='Set NumPy RNG seed value')
@click.option('--algo_kw', multiple=True, callback=options.callback_dict,
              help='Algorithm parameter overrides')
@click.option('--result_prefix', type=str, default='', show_default=True,
              multiple=True,
              help='Plot coef/rmse from refit that used this prefix')
@click.pass_context
def pixel(ctx, config, px, py, band, plot, ylim, style, cmap,
          embed, seed, algo_kw, result_prefix):
    # Set seed
    np.random.seed(seed)
    # Convert band to index
    band -= 1
    # Format result prefix
    if result_prefix:
        result_prefix = set((_pref if _pref[-1] == '_' else _pref + '_')
                         for _pref in result_prefix)
        result_prefix.add('')  # add in no prefix to show original fit
    else:
        result_prefix = ('')

    # Get colormap
    if cmap not in mpl.cm.cmap_d:
        raise click.ClickException('Cannot find specified colormap ({}) in '
                                   'matplotlib'.format(cmap))

    # Parse config
    cfg = parse_config_file(config)

    # Apply algorithm overrides
    for kw in algo_kw:
        value = yaml.load(algo_kw[kw])
        cfg = trawl_replace_keys(cfg, kw, value)
    if algo_kw:  # revalidate configuration
        cfg = convert_config(cfg)

    # Dataset information
    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              date_format=cfg['dataset']['date_format'])
    df['image_ID'] = get_image_IDs(df['filename'])
    df['x'] = df['date']
    dates = df['date'].values

    # Initialize timeseries model
    model = cfg['YATSM']['algorithm_cls']
    algo_cfg = cfg[cfg['YATSM']['algorithm']]
    yatsm = model(estimator=cfg['YATSM']['estimator'],
                  **algo_cfg.get('init', {}))
    yatsm.px = px
    yatsm.py = py

    # Setup algorithm and create design matrix (if needed)
    X = yatsm.setup(df, **cfg)
    design_info = getattr(X, 'design_info', None)

    # Read pixel data
    Y = read_pixel_timeseries(df['filename'], px, py)
    if Y.shape[0] != cfg['dataset']['n_bands']:
        raise click.ClickException(
            'Number of bands in image {f} ({nf}) do not match number in '
            'configuration file ({nc})'.format(
                f=df['filename'][0],
                nf=Y.shape[0],
                nc=cfg['dataset']['n_bands']))

    # Preprocess pixel data
    X, Y, dates = yatsm.preprocess(X, Y, dates, **cfg['dataset'])

    # Convert ordinal to datetime
    dt_dates = np.array([dt.datetime.fromordinal(d) for d in dates])

    # Plot before fitting
    with plt.xkcd() if style == 'xkcd' else mpl.style.context(style):
        for _plot in plot:
            if _plot == 'TS':
                plot_TS(dt_dates, Y[band, :])
            elif _plot == 'DOY':
                plot_DOY(dt_dates, Y[band, :], cmap)
            elif _plot == 'VAL':
                plot_VAL(dt_dates, Y[band, :], cmap)

            if ylim:
                plt.ylim(ylim)
            plt.title('Timeseries: px={px} py={py}'.format(px=px, py=py))
            plt.ylabel('Band {b}'.format(b=band + 1))
            plt.tight_layout()
            plt.show()

    # Fit model
    yatsm.fit(X, Y, dates, **algo_cfg.get('fit', {}))
    for prefix, estimator, stay_reg, fitopt in zip(
            cfg['YATSM']['refit']['prefix'],
            cfg['YATSM']['refit']['prediction_object'],
            cfg['YATSM']['refit']['stay_regularized'],
            cfg['YATSM']['refit']['fit']):
        yatsm.record = postprocess.refit_record(
            yatsm, prefix, estimator,
            fitopt=fitopt, keep_regularized=stay_reg)

    # Plot after predictions
    with plt.xkcd() if style == 'xkcd' else mpl.style.context(style):
            for _plot in plot:
                if _plot == 'TS':
                    plot_TS(dt_dates, Y[band, :])
                elif _plot == 'DOY':
                    plot_DOY(dt_dates, Y[band, :], cmap)
                elif _plot == 'VAL':
                    plot_VAL(dt_dates, Y[band, :], cmap)

                if ylim:
                    plt.ylim(ylim)
                plt.title('Timeseries: px={px} py={py}'.format(px=px, py=py))
                plt.ylabel('Band {b}'.format(b=band + 1))

                for _prefix in set(result_prefix):
                    plot_results(band, cfg, yatsm, design_info,
                                 result_prefix=_prefix,
                                 plot_type=_plot)

                plt.tight_layout()
                plt.show()

    if embed:
        console.open_interpreter(
            yatsm,
            message=("Additional functions:\n"
                     "plot_TS, plot_DOY, plot_VAL, plot_results"),
            variables={
                'config': cfg,
            },
            funcs={
                'plot_TS': plot_TS, 'plot_DOY': plot_DOY,
                'plot_VAL': plot_VAL, 'plot_results': plot_results
            }
        )


def plot_TS(dates, y):
    """ Create a standard timeseries plot

    Args:
        dates (iterable): sequence of datetime
        y (np.ndarray): variable to plot
    """
    # Plot data
    plt.scatter(dates, y, c='k', marker='o', edgecolors='none', s=35)
    plt.xlabel('Date')


def plot_DOY(dates, y, mpl_cmap):
    """ Create a DOY plot

    Args:
        dates (iterable): sequence of datetime
        y (np.ndarray): variable to plot
        mpl_cmap (colormap): matplotlib colormap
    """
    doy = np.array([d.timetuple().tm_yday for d in dates])
    year = np.array([d.year for d in dates])

    sp = plt.scatter(doy, y, c=year, cmap=mpl_cmap,
                     marker='o', edgecolors='none', s=35)
    plt.colorbar(sp)

    months = mpl.dates.MonthLocator()  # every month
    months_fmrt = mpl.dates.DateFormatter('%b')

    plt.tick_params(axis='x', which='minor', direction='in', pad=-10)
    plt.axes().xaxis.set_minor_locator(months)
    plt.axes().xaxis.set_minor_formatter(months_fmrt)

    plt.xlim(1, 366)
    plt.xlabel('Day of Year')


def plot_VAL(dates, y, mpl_cmap, reps=2):
    """ Create a "Valerie Pasquarella" plot (repeated DOY plot)

    Args:
        dates (iterable): sequence of datetime
        y (np.ndarray): variable to plot
        mpl_cmap (colormap): matplotlib colormap
        reps (int, optional): number of additional repetitions
    """
    doy = np.array([d.timetuple().tm_yday for d in dates])
    year = np.array([d.year for d in dates])

    # Replicate `reps` times
    _doy = doy.copy()
    for r in range(1, reps + 1):
        _doy = np.concatenate((_doy, doy + r * 366))
    _year = np.tile(year, reps + 1)
    _y = np.tile(y, reps + 1)

    sp = plt.scatter(_doy, _y, c=_year, cmap=mpl_cmap,
                     marker='o', edgecolors='none', s=35)
    plt.colorbar(sp)
    plt.xlabel('Day of Year')


def plot_results(band, cfg, model, design_info,
                 result_prefix='', plot_type='TS'):
    """ Plot model results

    Args:
        band (int): plot results for this band
        cfg (dict): YATSM configuration dictionary
        model (YATSM model): fitted YATSM timeseries model
        design_info (patsy.DesignInfo): patsy design information
        result_prefix (str): Prefix to 'coef' and 'rmse'
        plot_type (str): type of plot to add results to (TS, DOY, or VAL)
    """
    # Results prefix
    result_k = model.record.dtype.names
    coef_k = result_prefix + 'coef'
    rmse_k = result_prefix + 'rmse'
    if coef_k not in result_k or rmse_k not in result_k:
        raise KeyError('Cannot find result prefix "{}" in results'
                       .format(result_prefix))
    if result_prefix:
        click.echo('Using "{}" re-fitted results'.format(result_prefix))

    # Handle reverse
    step = -1 if cfg['YATSM']['reverse'] else 1

    # Remove categorical info from predictions
    design = re.sub(r'[\+\-][\ ]+C\(.*\)', '',
                    cfg['YATSM']['design_matrix'])

    i_coef = []
    for k, v in design_info.column_name_indexes.iteritems():
        if not re.match('C\(.*\)', k):
            i_coef.append(v)
    i_coef = np.asarray(i_coef)

    _prefix = result_prefix or cfg['YATSM']['prediction']
    for i, r in enumerate(model.record):
        label = 'Model {i} ({prefix})'.format(i=i, prefix=_prefix)
        if plot_type == 'TS':
            # Prediction
            mx = np.arange(r['start'], r['end'], step)
            mX = patsy.dmatrix(design, {'x': mx}).T

            my = np.dot(r[coef_k][i_coef, band], mX)
            mx_date = np.array([dt.datetime.fromordinal(int(_x)) for _x in mx])
            # Break
            if r['break']:
                bx = dt.datetime.fromordinal(r['break'])
                plt.axvline(bx, c='red', lw=2)

        elif plot_type in ('DOY', 'VAL'):
            yr_end = dt.datetime.fromordinal(r['end']).year
            yr_start = dt.datetime.fromordinal(r['start']).year
            yr_mid = int(yr_end - (yr_end - yr_start) / 2)

            mx = np.arange(dt.date(yr_mid, 1, 1).toordinal(),
                           dt.date(yr_mid + 1, 1, 1).toordinal(), 1)
            mX = patsy.dmatrix(design, {'x': mx}).T

            my = np.dot(r[coef_k][i_coef, band], mX)
            mx_date = np.array([dt.datetime.fromordinal(d).timetuple().tm_yday
                                for d in mx])

            label = 'Model {i} - {yr} ({prefix})'.format(i=i, yr=yr_mid,
                                                         prefix=_prefix)

        plt.plot(mx_date, my, lw=3, label=label)
    leg = plt.legend()
    leg.draggable(state=True)


# UTILITY FUNCTIONS
def trawl_replace_keys(d, key, value, s=''):
    """ Return modified dictionary ``d``
    """
    md = d.copy()
    for _key in md:
        if isinstance(md[_key], dict):
            # Recursively replace
            md[_key] = trawl_replace_keys(md[_key], key, value,
                                          s='{}[{}]'.format(s, _key))
        else:
            if _key == key:
                s += '[{}]'.format(_key)
                click.echo('Replacing d{k}={ov} with {nv}'
                           .format(k=s, ov=md[_key], nv=value))
                md[_key] = value
    return md
