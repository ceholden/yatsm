""" Command line interface for creating maps of YATSM algorithm output
"""
import logging

import click
import numpy as np
from osgeo import gdal

from . import options
from ..mapping import (get_classification, get_phenology,
                       get_coefficients, get_prediction)
from ..utils import write_output
from ..regression import design_coefs

gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')


@click.command(short_help='Make map of YATSM output for a given date')
@click.argument('map_type', metavar='<map_type>',
                type=click.Choice(['coef', 'predict', 'class', 'pheno']))
@options.arg_date()
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--warn-on-empty', is_flag=True,
              help='Warn user when reading in empty results files')
@click.option('--band', '-b', multiple=True, metavar='<band>', type=int,
              callback=options.valid_int_gt_zero,
              help='Bands to export for coefficient/prediction maps')
@click.option('--coef', '-c', multiple=True, metavar='<coef>',
              type=click.Choice(design_coefs), default=('all', ),
              help='Coefficients to export for coefficient maps')
@click.option('--after', is_flag=True,
              help='Use time segment after <date> if needed for map')
@click.option('--before', is_flag=True,
              help='Use time segment before <date> if needed for map')
@click.option('--qa', is_flag=True,
              help='Add QA band identifying segment type')
@click.option('--refit_prefix', default='', show_default=True,
              help='Use coef/rmse with refit prefix for coefficient/prediction'
                   ' maps')
@click.option('--amplitude', is_flag=True,
              help='Export amplitude of sin/cosine pairs instead of '
                   'individual coefficient estimates')
@click.option('--predict-proba', 'predict_proba', is_flag=True,
              help='Include prediction probability band (scaled by 10,000)')
@click.pass_context
def map(ctx, map_type, date, output,
        root, result, image, date_frmt, ndv, gdal_frmt, warn_on_empty,
        band, coef, after, before, qa, refit_prefix, amplitude, predict_proba):
    """
    Map types: coef, predict, class, pheno

    \b
    Map QA flags:
        - 0 => no values
        - 1 => before
        - 2 => after
        - 3 => intersect

    \b
    Examples:
    > yatsm map --coef intercept --coef slope
    ... --band 3 --band 4 --band 5 --ndv -9999
    ... coef 2000-01-01 coef_map.gtif

    \b
    > yatsm map -c intercept -c slope -b 3 -b 4 -b 5 --ndv -9999
    ... coef 2000-01-01 coef_map.gtif

    \b
    > yatsm map --date "%Y-%j" predict 2000-001 prediction.gtif

    \b
    > yatsm map --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

    \b
    Notes:
        - Image predictions will not use categorical information in timeseries
          models.
    """
    if len(band) == 0:
        band = 'all'

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except RuntimeError as err:
        raise click.ClickException('Could not open example image for reading '
                                   '(%s)' % str(err))

    date = date.toordinal()

    # Append underscore to prefix if not included
    if refit_prefix and not refit_prefix.endswith('_'):
        refit_prefix += '_'

    band_names = None
    if map_type == 'class':
        raster, band_names = get_classification(
            date, result, image_ds,
            after=after, before=before, qa=qa,
            pred_proba=predict_proba, warn_on_empty=warn_on_empty
        )
    elif map_type == 'coef':
        raster, band_names = get_coefficients(
            date, result, image_ds,
            band, coef,
            prefix=refit_prefix, amplitude=amplitude,
            after=after, before=before, qa=qa,
            ndv=ndv, warn_on_empty=warn_on_empty
        )
    elif map_type == 'predict':
        raster, band_names = get_prediction(
            date, result, image_ds,
            band,
            prefix=refit_prefix,
            after=after, before=before, qa=qa,
            ndv=ndv, warn_on_empty=warn_on_empty
        )
    elif map_type == 'pheno':
        raster, band_names = get_phenology(
            date, result, image_ds,
            after=after, before=before, qa=qa,
            ndv=ndv, warn_on_empty=warn_on_empty)

    write_output(raster, output, image_ds,
                 gdal_frmt, ndv, band_names)

    image_ds = None
