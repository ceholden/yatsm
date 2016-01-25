""" Command line interface for creating changemaps of YATSM algorithm output
"""
import logging
import os

import click
from osgeo import gdal

from . import options
from ..mapping import get_change_date, get_change_num
from ..utils import write_output

gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')


@click.command(
    short_help='Map change found by YATSM algorithm over time period')
@click.argument('map_type', metavar='<map_type>',
                type=click.Choice(['first', 'last', 'num']))
@options.arg_date(var='start_date', metavar='<start_date>')
@options.arg_date(var='end_date', metavar='<end_date>')
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--out_date', 'out_date_frmt', metavar='<format>',
              default='%Y%j', show_default=True, help='Output date format')
@click.option('--warn-on-empty', is_flag=True,
              help='Warn user when reading in empty results files')
@click.option('--magnitude', is_flag=True,
              help='Add magnitude of change as extra image '
                   '(pattern is [name]_mag[ext])')
@click.pass_context
def changemap(ctx, map_type, start_date, end_date, output,
              root, result, image, date_frmt, ndv, gdal_frmt, out_date_frmt,
              warn_on_empty, magnitude):
    """
    Examples: TODO
    """
    gdal_frmt = str(gdal_frmt)  # GDAL GetDriverByName doesn't work on Unicode

    frmt = '%Y%j'
    start_txt, end_txt = start_date.strftime(frmt), end_date.strftime(frmt)
    start_date, end_date = start_date.toordinal(), end_date.toordinal()

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    if map_type in ('first', 'last'):
        changemap, magnitudemap, magnitude_indices = get_change_date(
            start_date, end_date, result, image_ds,
            first=map_type == 'first', out_format=out_date_frmt,
            magnitude=magnitude,
            ndv=ndv, pattern='yatsm_r*', warn_on_empty=warn_on_empty
        )

        band_names = ['ChangeDate_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

        if magnitudemap is not None:
            band_names = (['Magnitude Index {}'.format(i) for
                           i in magnitude_indices])
            name, ext = os.path.splitext(output)
            output = name + '_mag' + ext
            write_output(magnitudemap, output, image_ds, gdal_frmt, ndv,
                         band_names=band_names)

    elif map_type == 'num':
        changemap = get_change_num(
            start_date, end_date, result, image_ds,
            ndv=ndv, pattern='yatsm_r*', warn_on_empty=warn_on_empty
        )

        band_names = ['NumChanges_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

    image_ds = None
