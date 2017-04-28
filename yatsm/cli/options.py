""" YATSM command line interface """
import datetime as dt
import functools
import logging
import os

import click
from rasterio.rio.options import bounds_opt as opt_bounds  # NOQA

from yatsm.executor import get_executor, EXECUTOR_DEFAULTS, EXECUTOR_TYPES

logger = logging.getLogger(__name__)


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


def callback_nodata(ctx, param, value):
    """ Nodata handling
    """
    if not value or value.lower() in ('none', 'null', 'no', 'na', 'nan'):
        return None
    else:
        try:
            return float(value)
        except:
            raise click.BadParameter('{!r} is not a number'.format(value),
                                     param=param, param_hint='nodata')


# Config handling
def fetch_config(ctx):
    """ Fetch ``config`` option and return :ref:`yatsm.api.Config`
    """
    config_file = ctx.obj and ctx.obj.get('config', None)
    if not config_file:
        _opts = dict((o.name, o) for o in ctx.parent.command.params)
        raise click.BadParameter('Must specify configuration file',
                                 ctx=ctx.parent, param=_opts['config'])

    from yatsm.api import Config
    try:
        config = Config.from_file(config_file)
    except Exception as err:
        logger.exception('Cannot parse config file: "{0}"'.format(config_file))
        raise click.BadParameter('Cannot parse config file %s' % config_file)
    else:
        return config


opt_config = click.option(
    '--config', '-C',
    type=click.Path(exists=True, readable=True,
                    dir_okay=False, resolve_path=True),
    default=lambda: os.environ.get('YATSM_CONFIG', None),
    allow_from_autoenv=True,
    nargs=1,
    help='YATSM configuration file [default: ${YATSM_CONFIG}]'
)


# ARGUMENTS
def _arg_date(name, date_format_key='date_format', **kwds):
    def _arg_date(f):
        def callback(ctx, param, value):
            try:
                value = dt.datetime.strptime(value,
                                             ctx.params[date_format_key])
            except KeyError:
                raise click.ClickException(
                    'Need to use `date_format_opt` when using `date_arg`')
            except ValueError:
                raise click.BadParameter(
                    'Cannot parse {v} to date with format {f}'.format(
                        v=value, f=ctx.params['date_format']))
            else:
                return value
        return click.argument(name, callback=callback, **kwds)(f)
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

    return click.argument('job_number', nargs=1, callback=callback)(f)


arg_output = click.argument(
    'output',
    type=click.Path(writable=True, dir_okay=False,
                    resolve_path=True))


arg_total_jobs = click.argument(
    'total_jobs',
    nargs=1,
    type=click.INT
)


arg_date = _arg_date('date', date_format_key='date_format')
arg_start_date = _arg_date('start_date', date_format_key='date_format')
arg_end_date = _arg_date('end_date', date_format_key='date_format')


# OPTIONS
opt_date_format = click.option(
    '--date_format', '--dformat', 'date_format',
    default='%Y-%m-%d',
    show_default=True,
    is_eager=True,
    help='Input date format')


def opt_map_date_format(f):
    def callback(ctx, param, value):
        # TODO: this needs to ensure conversion if required, because
        #       results dates eventually won't always be in ordinal
        #       date format (in fact, it makes little sense because
        #       basically nothing in Python uses it, except certain
        #       regression models for historic reasons)
        #       See: `yatsm.tslib.datetime2int`
        if value.lower() == 'ordinal':
            return 'ordinal'
        elif isinstance(value, str):
            test_dt = dt.datetime.now()
            try:
                int(test_dt.strftime(value))
            except Exception as exc:
                msg = ('"{0}" is not a valid date format string for '
                       'parameter "{1}": {2!r}'
                       .format(value, param.metavar, exc))
                raise click.BadParameter(msg)
            else:
                return value
        raise click.BadParameter('{0} must be a valid `datetime.strftime` '
                                 'format string or "ordinal"'
                                 .format(param.metavar))
    return click.option(
        '--map_date_format', '--mdformat', 'map_date_format',
        callback=callback,
        default='%Y%m%d', show_default=True,
        help='Output map date format ("ordinal" or a date fstring)')(f)


opt_format = click.option(
    '-f', '--format', '--driver',
    default='GTiff', show_default=True,
    help="Output format driver")


opt_creation_options = click.option(
    '--co', '--profile', 'creation_options',
    metavar='NAME=VALUE',
    multiple=True,
    callback=callback_dict,
    show_default=True,
    help="Driver specific creation options."
         "See the documentation for the selected output driver for "
         "more information.")


opt_force_overwrite = click.option(
    '--force-overwrite', 'force_overwrite',
    is_flag=True, type=bool, default=False, show_default=True,
    help="Always overwrite an existing output file.")


opt_nodata = click.option(
    '--nodata', callback=callback_nodata,
    default='-9999', show_default=True,
    metavar='NUMBER|nan', help="Set a Nodata value.")


# COMBINED DECORATOR SHORTCUTS
def _combined_decorations(func, optargs=None):
    optargs = optargs or []

    new_func = func
    for f in optargs:
        new_func = f(new_func)

    @functools.wraps(new_func)
    def decorator(*args, **kwargs):
        return new_func(*args, **kwargs)

    return decorator


# TODO: --bounds, --like, etc?
#: tuple: Collection of optional arguments used in map scripts
opts_mapping = (
    click.pass_context,  # ctx
    opt_force_overwrite,  # force_overwrite
    opt_format,  # driver
    opt_nodata,  # nodata
    opt_creation_options,  # co
)


#: callable: Decorator that combines all Click options from :ref:`opts_mapping`
mapping_decorations = functools.partial(_combined_decorations,
                                        optargs=opts_mapping)


# SEGMENT HANDLING
# Use segment after DATE
opt_after = click.option('--after', is_flag=True,
                         help='Use time segment after DATE if needed')


# Use segment before DATE
opt_before = click.option('--before', is_flag=True,
                          help='Use time segment before DATE if needed')


# Output QA/QC band?
opt_qa_band = click.option('--qa', is_flag=True,
                           help='Add QA band identifying segment type')


# EXECUTOR
opt_executor = click.option(
    '--executor',
    default=(EXECUTOR_TYPES[0], None), show_default=True,
    type=(click.Choice(EXECUTOR_TYPES), str),
    help='Pipeline executor (e.g., {})'
         .format(', '.join(['"%s %s"' % (k, v) for k, v
                 in EXECUTOR_DEFAULTS.items()])),
    callback=lambda ctx, param, value: get_executor(*value)
)
