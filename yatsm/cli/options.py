""" YATSM command line interface """
from datetime import datetime as dt
import functools
import os

import click
import cligj
from rasterio.rio import options as rio_options

from yatsm.executor import get_executor, EXECUTOR_DEFAULTS, EXECUTOR_TYPES


# CLI VALIDATORS
def valid_int_gt_zero(ctx, param, value):
    """ Validator for integers > 0 (value >= 1)"""
    def _validator(param, value):
        try:
            value = int(value)
        except Exception as e:
            raise click.BadParameter('%s must be integer above zero: %s'
                                     % (param.metavar, e.message))
        if value <= 0:
            raise click.BadParameter('%s must be an integer above zero'
                                     % param.metavar)
        return value

    if param.multiple:
        return [_validator(param, v) for v in value]
    else:
        return _validator(param, value)


# ARGUMENTS
arg_config_file = click.argument(
    'configfile',
    nargs=1,
    type=click.Path(exists=True, readable=True,
                    dir_okay=False, resolve_path=True),
    metavar='<config>')


arg_output = click.argument(
    'output',
    metavar='<output>',
    type=click.Path(writable=True, dir_okay=False,
                    resolve_path=True))


arg_total_jobs = click.argument(
    'total_jobs',
    nargs=1,
    type=click.INT,
    metavar='<total_jobs>')


def arg_date(var='date', metavar='<date>', date_frmt_key='date_frmt'):
    def _arg_date(f):
        def callback(ctx, param, value):
            try:
                value = dt.strptime(value, ctx.params[date_frmt_key])
            except KeyError:
                raise click.ClickException(
                    'Need to use `date_format_opt` when using `date_arg`')
            except ValueError:
                raise click.BadParameter(
                    'Cannot parse {v} to date with format {f}'.format(
                        v=value, f=ctx.params['date_frmt']))
            else:
                return value
        return click.argument(var, metavar=metavar, callback=callback)(f)
    return _arg_date


def arg_job_number(f):
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


# OPTIONS
opt_date_format = click.option(
    '--date', 'date_frmt',
    default='%Y-%m-%d',
    metavar='<format>',
    show_default=True,
    is_eager=True,
    help='Input date format')


# EXECUTOR
opt_executor = click.option(
    '--executor',
    default=(EXECUTOR_TYPES[0], None), show_default=True,
    type=(click.Choice(EXECUTOR_TYPES), str),
    help='Pipeline executor (e.g., {})'.format(
            ', '.join(['"%s %s"' % (k, v) for k, v
            in EXECUTOR_DEFAULTS.items()
        ])
    ),
    callback=get_executor
)



# CALLBACKS
def callback_dict(ctx, param, value):
    """ Call back for dict style arguments (e.g., KEY=VALUE)
    """
    if not value:
        return {}
    else:
        d = {}
        for val in value:
            if '=' not in val:
                raise click.BadParameter(
                    'Must specify {p} as KEY=VALUE ({v} given)'.format(
                        p=param, v=value))
            else:
                k, v = val.split('=', 1)
                d[k] = v
        return d
