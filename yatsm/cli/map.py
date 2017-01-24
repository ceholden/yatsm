""" Command line interface for creating maps of YATSM algorithm output
"""
import functools
import logging

import click
import cligj
from rasterio.rio import options as rio_options

from . import options
from ..regression.design import design_coefs

logger = logging.getLogger('yatsm')

opt_after = click.option('--after', is_flag=True,
                         help='Use time segment after <date> if needed for map')
opt_before = click.option('--before', is_flag=True,
                          help='Use time segment before <date> if needed for map')
opt_qa_band = click.option('--qa', is_flag=True,
                           help='Add QA band identifying segment type')


@click.group(short_help='Make map of YATSM output for a given date')
@click.pass_context
def map(ctx):
    """ Make maps
    """
    click.echo('Map')


@map.command(short_help='Coefficient map')
@options.mapping_decorations
def coef(ctx, driver, nodata, co, force_overwrite):
    """ Coefficient maps
    """
    click.echo("Coefficient map")


@map.command(short_help='Synthetic prediction map')
@options.mapping_decorations
def predict(ctx, driver, nodata, co, force_overwrite):
    """ Synthetic prediction map
    """
    click.echo("Synthetic prediction maps")


@map.command(name='class', short_help='Classification map')
@options.mapping_decorations
def class_(ctx, driver, nodata, co, force_overwrite):
    """ Classification map
    """
    click.echo("Classification maps")


@map.command(short_help='Phenology')
@options.mapping_decorations
def pheno(ctx, driver, nodata, co, force_overwrite):
    """ Phenology map
    """
    click.echo("Phenology maps")
