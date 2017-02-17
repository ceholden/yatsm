""" YATSM command line interface """
from datetime import datetime as dt
import functools
import logging

import click

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


# ARGUMENTS
def _callback_arg_config(ctx, param, value):
    from yatsm.api import Config
    try:
        config = Config.from_file(value)
    except Exception as err:
        logger.exception('Cannot parse config file %s' % value, err)
        click.BadParameter('Cannot parse config file %s' % value)
        raise
    else:
        return config


def _arg_date(name, date_format_key='date_format', **kwds):
    def _arg_date(f):
        def callback(ctx, param, value):
            try:
                value = dt.strptime(value, ctx.params[date_format_key])
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


arg_config = click.argument(
    'config',
    nargs=1,
    type=click.Path(exists=True, readable=True,
                    dir_okay=False, resolve_path=True),
    callback=_callback_arg_config)


arg_output = click.argument(
    'output',
    type=click.Path(writable=True, dir_okay=False,
                    resolve_path=True))


arg_total_jobs = click.argument(
    'total_jobs',
    nargs=1,
    type=click.INT)


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
