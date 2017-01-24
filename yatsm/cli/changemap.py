""" Command line interface for creating changemaps of YATSM algorithm output
"""
import logging
import os

import click


from . import options


logger = logging.getLogger('yatsm')


@click.group(short_help='Map change found by YATSM algorithm over time period')
@click.pass_context
def changemap(ctx):
    pass


@changemap.command(short_help="Date of first change")
@options.mapping_decorations
def first(ctx, driver, nodata, co, force_overwrite):
    """ Date of first change
    """
    click.echo("First change detected")


@changemap.command(short_help="Date of last change")
@options.mapping_decorations
def last(ctx, driver, nodata, co, force_overwrite):
    """ Date of last change
    """
    click.echo("Last change detected")


@changemap.command(short_help="Number of changes")
@options.mapping_decorations
def num(ctx, driver, nodata, co, force_overwrite):
    """ Number of changes
    """
    click.echo("Number of changes")
