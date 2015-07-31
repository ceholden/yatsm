""" Command line interface for creating maps of YATSM algorithm output
"""
import datetime as dt
import logging
import os
import re

import click
import numpy as np
from osgeo import gdal
import patsy

from yatsm.cli.cli import (cli, date_arg, date_format_opt,
                           rootdir_opt, resultdir_opt, exampleimg_opt)
from yatsm.utils import find_results, iter_records, write_output
from yatsm.regression import design_to_indices, design_coefs
from yatsm.regression.transforms import harm

gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

# QA/QC values for segment types
_intersect_qa = 3
_after_qa = 2
_before_qa = 1

# Filters for results
_result_record = 'yatsm_r*'
# number of days in year
_days = 365.25
w = 2 * np.pi / _days

WARN_ON_EMPTY = False


@cli.command(short_help='Make map of YATSM output for a given date')
@click.argument('type', metavar='<type>',
                type=click.Choice(['coef', 'predict', 'class', 'pheno']))
@date_arg
@click.argument('output', metavar='<output>',
                type=click.Path(writable=True, dir_okay=False,
                                resolve_path=True))
@rootdir_opt
@resultdir_opt
@exampleimg_opt
@date_format_opt
@click.option('--warn-on-empty', is_flag=True,
              help='Warn user when reading in empty results files')
@click.option('--after', is_flag=True,
              help='Use time segment after <date> if needed for map')
@click.option('--before', is_flag=True,
              help='Use time segment before <date> if needed for map')
@click.option('--qa', is_flag=True,
              help='Add QA band identifying segment type')
@click.option('--predict-proba', 'predict_proba', is_flag=True,
              help='Include prediction probability band (scaled by 10,000)')
@click.option('--band', '-b', multiple=True, metavar='<band>',
              help='Bands to export for coefficient/prediction maps')
@click.option('--robust', is_flag=True,
              help='Use robust results for coefficient/prediction maps')
@click.option('--coef', '-c', multiple=True, metavar='<coef>',
              help='Coefficients to export for coefficient maps')
@click.pass_context
def map(ctx, type, date, output,
        root, result, image, date_frmt, warn_on_empty,
        after, before, qa, predict_proba, band, robust, coef):
    """
    Map types: coef, predict, class, pheno

    Map QA flags:
        - 1 => before
        - 2 => after
        - 3 => intersect

    Examples:
    > yatsm map --coef "intercept, slope" --band "3, 4, 5" --ndv -9999 coef
    ... 2000-01-01 coef_map.gtif

    > yatsm map --date "%Y-%j" predict 2000-001 prediction.gtif

    > yatsm map --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

    Notes:
        - Image predictions will not use categorical information in timeseries
          models.
    """
    raise NotImplementedError('CLI in place; TODO actual script')
