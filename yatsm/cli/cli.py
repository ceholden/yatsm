""" YATSM command line interface """
import logging

import click

import yatsm


# Logging config
FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')


# YATSM CLI group
@click.group(help='YATSM command line interface')
@click.version_option(yatsm.version.__version__)
@click.option('--verbose', '-v', is_flag=True, count=True, help='Be verbose')
@click.option('--quiet', '-q', is_flag=True, count=True, help='Be quiet')
@click.pass_context
def cli(ctx, verbose, quiet):
    if verbose:
        logger.setLevel(logging.DEBUG)
    if quiet:
        logger.setLevel(logging.WARNING)

    pass


# TESTING -- DELETE LATER
if __name__ == '__main__':
    cli()
