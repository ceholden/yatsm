""" Command line interface for running YATSM algorithms on individual pixels
"""
from datetime import datetime as dt
import logging

try:
    from IPython import embed as IPython_embed
    has_embed = True
except:
    has_embed = False

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
import patsy

from yatsm.cli.cli import cli, config_file_arg
from yatsm.config_parser import parse_config_file
import yatsm._cyprep as cyprep
from yatsm.utils import csvfile_to_dataset
from yatsm.reader import read_pixel_timeseries
from yatsm.regression.transforms import harm
from yatsm.yatsm import YATSM

avail_plots = ['TS', 'DOY']

plot_styles = []
if hasattr(mpl, 'style'):
    plot_styles = mpl.style.available
if hasattr(plt, 'xkcd'):
    plot_styles.append('xkcd')

logger = logging.getLogger('yatsm')


@cli.command(short_help='Run YATSM algorithm on individual pixels')
@config_file_arg
@click.argument('px', metavar='<px>', nargs=1, type=click.INT)
@click.argument('py', metavar='<py>', nargs=1, type=click.INT)
@click.option('--band', metavar='<n>', nargs=1, type=click.INT, default=1,
              help='Band to plot')
@click.option('--plot', default=('TS',), multiple=True,
              type=click.Choice(avail_plots), help='Plot type')
@click.option('--ylim', metavar='<min> <max>', nargs=2, type=float,
              help='Y-axis limits')
@click.option('--style', metavar='<style>', default='ggplot',
              type=click.Choice(plot_styles), help='Plot style')
@click.option('--embed', is_flag=True,
              help='Drop to embedded IPython shell at various points')
@click.pass_context
def pixel(ctx, config, px, py, band, plot, ylim, style, embed):
    # Convert band to index
    band -= 1

    # Parse config
    dataset_config, yatsm_config = parse_config_file(config)

    # Locate and fetch attributes from data
    dataset = csvfile_to_dataset(dataset_config['input_file'],
                                 date_format=dataset_config['date_format'])
    dates = dataset['dates']
    sensors = dataset['sensors']
    images = dataset['images']

    # Read in data and setup Y and X data
    Y = read_pixel_timeseries(images, px, py)
    X = patsy.dmatrix(yatsm_config['design_matrix'],
                      {'x': dates, 'sensor': sensors})

    # Mask out of range data
    valid = cyprep.get_valid_mask(Y[:dataset_config['mask_band'] - 1, :],
                                  dataset_config['min_values'],
                                  dataset_config['max_values'])
    # Add mask band to mask and remove from Y
    valid = (valid * np.in1d(Y[dataset_config['mask_band'] - 1, :],
                             dataset_config['mask_values'],
                             invert=True)).astype(np.bool)
    Y = np.delete(Y, dataset_config['mask_band'] - 1, axis=0)

    # Apply mask
    dates = np.array([dt.fromordinal(d) for d in dates[valid]])
    Y = Y[:, valid]
    X = X[valid, :]

    # Plot before fitting
    with plt.xkcd() if style == 'xkcd' else mpl.style.context(style):
        for _plot in plot:
            if _plot == 'TS':
                plot_TS(dates, Y[band, :])
            elif _plot == 'DOY':
                plot_DOY(dates, Y[band, :])
            plt.title('Timeseries: px={px} py={py}'.format(px=px, py=py))
            plt.ylabel('Band {b}'.format(b=band + 1))
            if embed and has_embed:
                IPython_embed('About to show {p} plot'.format(p=_plot))
            plt.show()


def plot_TS(dates, y):
    plt.plot(dates, y, 'ro')
    plt.xlabel('Date')


def plot_DOY(dates, y):
    doy = np.array([d.timetuple().tm_yday for d in dates])
    plt.plot(doy, y, 'ro')
    plt.xlabel('Day of Year')
