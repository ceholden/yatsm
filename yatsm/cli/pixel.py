""" Command line interface for running YATSM algorithms on individual pixels
"""
import datetime as dt
import logging
import re

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
import patsy
import yaml

from yatsm.algorithms import postprocess  # TODO: implement postprocessors
from yatsm.cli import options, console
from yatsm.config_parser import convert_config, parse_config_file
from yatsm import _cyprep as cyprep
from yatsm.utils import csvfile_to_dataframe, get_image_IDs
from yatsm.reader import read_pixel_timeseries
from yatsm.regression.transforms import harm  # noqa

avail_plots = ['TS', 'DOY', 'VAL']

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
@click.option('--cmap', metavar='<cmap>', default='perceptual_rainbow_16',
              show_default=True, help='DOY plot colormap')
@click.option('--embed', is_flag=True,
              help='Drop to (I)Python interpreter at various points')
@click.option('--seed', help='Set NumPy RNG seed value')
@click.option('--algo_kw', multiple=True, callback=options.callback_dict,
              help='Algorithm parameter overrides')
@click.pass_context
def pixel(ctx, config, px, py, band, plot, ylim, style, cmap,
          embed, seed, algo_kw):
    # Set seed
    np.random.seed(seed)
    # Convert band to index
    band -= 1

    # Get colormap
    if hasattr(palettable.colorbrewer, cmap):
        mpl_cmap = getattr(palettable.colorbrewer, cmap).mpl_colormap
    elif hasattr(palettable.cubehelix, cmap):
        mpl_cmap = getattr(palettable.cubehelix, cmap).mpl_colormap
    elif hasattr(palettable.wesanderson, cmap):
        mpl_cmap = getattr(palettable.wesanderson, cmap).mpl_colormap
    else:
        click.secho('Cannot find specified colormap in `palettable`', fg='red')
        raise click.Abort()

    # Parse config
    cfg = parse_config_file(config)

    # Apply algorithm overrides
    revalidate = False
    for kw in algo_kw:
        for cfg_key in cfg:
            if kw in cfg[cfg_key]:
                # Parse as YAML for type conversions used in config parser
                value = yaml.load(algo_kw[kw])

                print('Overriding cfg[%s][%s]=%s with %s' %
                      (cfg_key, kw, cfg[cfg_key][kw], value))
                cfg[cfg_key][kw] = value
                revalidate = True

    if revalidate:
        cfg = convert_config(cfg)

    # Locate and fetch attributes from data
    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              date_format=cfg['dataset']['date_format'])
    df['image_ID'] = get_image_IDs(df['filename'])

    # Setup X/Y
    df['x'] = df['date']
    X = patsy.dmatrix(cfg['YATSM']['design_matrix'], data=df)
    design_info = X.design_info

    Y = read_pixel_timeseries(df['filename'], px, py)
    if Y.shape[0] != cfg['dataset']['n_bands']:
        logger.error('Number of bands in image %s (%i) do not match number '
                     'in configuration file (%i)' %
                     (df['filename'][0], Y.shape[0],
                      cfg['dataset']['n_bands']))
        raise click.Abort()

    # Mask out of range data
    idx_mask = cfg['dataset']['mask_band'] - 1
    valid = cyprep.get_valid_mask(Y,
                                  cfg['dataset']['min_values'],
                                  cfg['dataset']['max_values']).astype(np.bool)
    valid *= np.in1d(Y[idx_mask, :], cfg['dataset']['mask_values'],
                     invert=True).astype(np.bool)

    # Apply mask
    Y = np.delete(Y, idx_mask, axis=0)[:, valid]
    X = X[valid, :]
    dates = np.array([dt.datetime.fromordinal(d) for d in df['date'][valid]])

    # Plot before fitting
    with plt.xkcd() if style == 'xkcd' else mpl.style.context(style):
        for _plot in plot:
            if _plot == 'TS':
                plot_TS(dates, Y[band, :])
            elif _plot == 'DOY':
                plot_DOY(dates, Y[band, :], mpl_cmap)
            elif _plot == 'VAL':
                plot_VAL(dates, Y[band, :], mpl_cmap)

            if ylim:
                plt.ylim(ylim)
            plt.title('Timeseries: px={px} py={py}'.format(px=px, py=py))
            plt.ylabel('Band {b}'.format(b=band + 1))
            plt.tight_layout()
            plt.show()

    # Eliminate config parameters not algorithm and fit model
    algo_cfg = cfg[cfg['YATSM']['algorithm']]
    yatsm = cfg['YATSM']['algorithm_cls'](
        estimator=cfg['YATSM']['prediction_object'],
        **algo_cfg.get('init', {}))
    yatsm.px = px
    yatsm.py = py
    yatsm.fit(X, Y, np.asarray(df['date'][valid]), **algo_cfg.get('fit', {}))

    # Plot after predictions
    with plt.xkcd() if style == 'xkcd' else mpl.style.context(style):
        for _plot in plot:
            if _plot == 'TS':
                plot_TS(dates, Y[band, :])
            elif _plot == 'DOY':
                plot_DOY(dates, Y[band, :], mpl_cmap)
            elif _plot == 'VAL':
                plot_VAL(dates, Y[band, :], mpl_cmap)

            if ylim:
                plt.ylim(ylim)
            plt.title('Timeseries: px={px} py={py}'.format(px=px, py=py))
            plt.ylabel('Band {b}'.format(b=band + 1))

            plot_results(band, cfg, yatsm, design_info, plot_type=_plot)

            plt.tight_layout()
            plt.show()

    if embed:
        console.open_interpreter(
            yatsm,
            message=("Additional functions:\n"
                     "plot_TS, plot_DOY, plot_VAL, plot_results"),
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


def plot_results(band, cfg, model, design_info, plot_type='TS'):
    """ Create a DOY plot

    Args:
        band (int): plot results for this band
        cfg (dict): YATSM configuration dictionary
        model (YATSM model): fitted YATSM timeseries model
        design_info (patsy.DesignInfo): patsy design information
        plot_type (str): type of plot to add results to (TS, DOY, or VAL)
    """
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

    for i, r in enumerate(model.record):
        label = 'Model {i}'.format(i=i)
        if plot_type == 'TS':
            # Prediction
            mx = np.arange(r['start'], r['end'], step)
            mX = patsy.dmatrix(design, {'x': mx}).T

            my = np.dot(r['coef'][i_coef, band], mX)
            mx_date = np.array([dt.datetime.fromordinal(int(_x)) for _x in mx])
            # Break
            if r['break']:
                bx = dt.datetime.fromordinal(r['break'])
                plt.axvline(bx, c='red', lw=2)

        elif plot_type == 'DOY':
            yr_end = dt.datetime.fromordinal(r['end']).year
            yr_start = dt.datetime.fromordinal(r['start']).year
            yr_mid = int(yr_end - (yr_end - yr_start) / 2)

            mx = np.arange(dt.date(yr_mid, 1, 1).toordinal(),
                           dt.date(yr_mid + 1, 1, 1).toordinal(), 1)
            mX = patsy.dmatrix(design, {'x': mx}).T

            my = np.dot(r['coef'][i_coef, band], mX)
            mx_date = np.array([dt.datetime.fromordinal(d).timetuple().tm_yday
                                for d in mx])

            label = 'Model {i} - {yr}'.format(i=i, yr=yr_mid)

        plt.plot(mx_date, my, lw=2, label=label)
    leg = plt.legend()
    leg.draggable(state=True)


def plot_lasso_debug(model):
    """ See example:
    http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
    """
    m_log_alphas = -np.log10(model.alphas_)
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='alpha: CV estimate')
    plt.xlabel('-log(alpha)')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: coordinate descent')


# UTILITY FUNCTIONS
def type_convert(value, example):
    """ Convert value (str) to dtype of `example`

    Args:
      value (str): string value to convert type
      example (int, float, bool, list, tuple, np.ndarray, etc.): `value`
        converted to type of `example` variable

    """
    dtype = type(example)
    if dtype is int:
        return int(value)
    elif dtype is float:
        return float(value)
    elif dtype in (list, tuple, np.ndarray):
        _dtype = type(example[0])
        return np.array([_dtype(v) for v in value.replace(',', ' ').split(' ')
                         if v])
    elif dtype is bool:
        if value.lower()[0] in ('t', 'y'):
            return True
        else:
            return False
