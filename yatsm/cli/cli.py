""" YATSM command line interface """
import logging
import os

import click

import yatsm

# Logging config
FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

_context = dict(
    help_option_names=['--help', '-h']
)


# YATSM CLI group
@click.group(help='YATSM command line interface', context_settings=_context)
@click.version_option(yatsm.__version__)
@click.option('--verbose', '-v', is_flag=True, help='Be verbose')
@click.option('--quiet', '-q', is_flag=True, help='Be quiet')
@click.pass_context
def cli(ctx, verbose, quiet):
    if verbose:
        logger.setLevel(logging.DEBUG)
    if quiet:
        logger.setLevel(logging.WARNING)


# CLI validators
def valid_band(ctx, param, value):
    """ Check image band validity (band >= 1)"""
    try:
        band = int(value)
        assert band >= 1
    except:
        raise click.BadParameter('Band must be integer above 1')
    return band

# CLI arguments and options
config_file_arg = click.argument(
    'config',
    nargs=1,
    type=click.Path(exists=True, readable=True,
                    dir_okay=False, resolve_path=True),
    metavar='<config>')

def job_number_arg(f):
    def callback(ctx, param, value):
        try:
            value = int(value)
        except:
            raise click.BadParameter('Must specify an integer >= 0')

        if value < 0:
            raise click.BadParameter('Must specify an integer >= 0')
        elif value == 0:
            return value
        else:
            return value - 1

    return click.argument('job_number', nargs=1, callback=callback,
                          metavar='<job_number>')(f)

total_jobs_arg = click.argument(
    'total_jobs',
    nargs=1,
    type=click.INT,
    metavar='<total_jobs>')

format_opt = click.option(
    '-f', '--format',
    default='GTiff',
    metavar='<driver>',
    help='Output format driver')

rootdir_opt = click.option(
    '--root',
    default='./',
    metavar='<directory>',
    help='Root timeseries directory',
    type=click.Path(exists=True, file_okay=False,
                    readable=True, resolve_path=True))


def resultdir_opt(f):
    def callback(ctx, param, value):
        # Check if path qualifies alone
        if os.path.isdir(value):
            _value = value
        else:
            # Check if path relative to root qualifies
            _value = os.path.join(ctx.params['root'], value)
            if not os.path.isdir(_value):
                raise click.BadParameter('Cannot find result directory '
                                         '"{d}"'.format(d=value))
        if not os.access(_value, os.R_OK):
            raise click.BadParameter('Found result directory but cannot '
                                     'read from "{d}"'.format(d=_value))
        return os.path.abspath(_value)
    return click.option('--result', '-r',
                        default='YATSM',
                        metavar='<directory>',
                        help='Directory of results',
                        callback=callback)(f)


def exampleimg_opt(f):
    def callback(ctx, param, value):
        # Check if file qualifies alone
        if os.path.isfile(value):
            _value = value
        else:
            # Check if path relative to root qualifies
            _value = os.path.join(ctx.params['root'], value)
            if not os.path.isfile(_value):
                raise click.BadParameter('Cannot find example image '
                                         '"{f}"'.format(f=value))
            if not os.access(_value, os.R_OK):
                raise click.BadParameter('Found example image but cannot '
                                         'read from "{f}"'.format(f=_value))
        return os.path.abspath(_value)
    return click.option('--image', '-i',
                        default='example_img',
                        metavar='<image>',
                        help='Example timeseries image',
                        callback=callback)(f)
