""" Command line interface for creating changemaps of YATSM algorithm output
"""
import logging
import os

import click

from . import options
from ..config import validate_and_parse_configfile


logger = logging.getLogger('yatsm')


opt_table = click.option('--table',
                         type=str, default=None, show_default=True,
                         help='Extract data from this table')


opt_column = click.option('--column',
                          type=str, default=None, show_default=True,
                          help='Extract data from this column')


@click.group(short_help='Map change found by YATSM algorithm over time period')
@click.pass_context
def changemap(ctx):
    pass


@changemap.command(short_help="Date of first change")
@options.arg_config
@options.arg_start_date
@options.arg_end_date
@options.arg_output
@opt_table
@options.opt_bounds
@options.mapping_decorations
@options.opt_date_format
def first(ctx, config, start_date, end_date, output,
          table, bounds,
          driver, nodata, creation_options, force_overwrite, date_format):
    """ Date of first change
    """
    import numpy as np
    import rasterio
    import tables as tb

    results = config.find_results(**config.results)  # TODO: extent?

    # TODO: manually specify extent (e.g., subset)
    crs = config.primary_reader.crs
    transform = config.primary_reader.transform
    bounds = config.primary_reader.bounds
    window = rasterio.windows.from_bounds(*bounds,
                                          transform=transform,
                                          boundless=True)
    kwds = {
        'driver': driver,
        'nodata': nodata,
        'dtype': np.int32,
        'count': 1,  # TODO: qa/qc
        'crs': crs,
        'height': window[0][1] - window[0][0],
        'width': window[1][1] - window[1][0],
        'transform': rasterio.windows.transform(window, transform)
    }
    kwds.update(**creation_options)

    columns = ('px', 'py', 'break_day', )

    with rasterio.open(output, 'w', **kwds) as dst:
        out = np.full(dst.shape, dst.nodata, dst.dtypes[0])
        for _result in results:
            try:
                with _result as result:  # Open
                    if not table:  # set if first time
                        _tables = list(result.tables())
                        if _tables:
                            table = _tables[0][0]
                            logger.info('Assuming you want table {}'
                                        .format(table))
                    segs = result.query(table, columns=columns,
                                        d_start=start_date, d_end=end_date)
                    y, x = rasterio.transform.rowcol(result.transform,
                                                     segs['px'], segs['py'])
                    out[y, x] = segs[columns[-1]]
            except tb.exceptions.HDF5ExtError as err:
                logger.error('Result file {} is corrupt or unreadable'
                             .format(_result.filename), err)

        from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA
        dst.write(out, 1)
        print(dst.tags())

    if not results:
        logger.exception('No results found in %s' % config.results['output'])
        raise click.Abort()


@changemap.command(short_help="Date of last change")
@options.mapping_decorations
def last(ctx, driver, nodata, creation_options, force_overwrite):
    """ Date of last change
    """
    click.echo("Last change detected")


@changemap.command(short_help="Number of changes")
@options.mapping_decorations
def num(ctx, driver, nodata, creation_options, force_overwrite):
    """ Number of changes
    """
    click.echo("Number of changes")
