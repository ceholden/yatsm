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

from yatsm.cli import options
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
@click.option('--band', '-b', multiple=True, metavar='<band>',
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
        - 1 => before
        - 2 => after
        - 3 => intersect

    \b
    Examples:
    > yatsm map --coef "intercept, slope" --band "3, 4, 5" --ndv -9999 coef
    ... 2000-01-01 coef_map.gtif

    > yatsm map --date "%Y-%j" predict 2000-001 prediction.gtif

    > yatsm map --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

    Notes:
        - Image predictions will not use categorical information in timeseries
          models.
    """
    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    # Append underscore to prefix if not included
    if refit_prefix and not refit_prefix.endswith('_'):
        refit_prefix += '_'

    band_names = None
    if map_type == 'class':
        raster, band_names = get_classification(
            date, result, image_ds,
            after=after, before=before, qa=qa,
            pred_proba=predict_proba
        )
    elif map_type == 'coef':
        raster, band_names = get_coefficients(
            date, result, image_ds,
            band, coef,
            prefix=refit_prefix, amplitude=amplitude,
            after=after, before=before, qa=qa,
            ndv=ndv
        )
    elif map_type == 'predict':
        raster, band_names = get_prediction(
            date, result, image_ds,
            band,
            prefix=refit_prefix,
            after=after, before=before, qa=qa,
            ndv=ndv
        )
    elif map_type == 'pheno':
        raster, band_names = get_phenology(
            date, result, image_ds,
            after=after, before=before, qa=qa,
            ndv=ndv)

    write_output(raster, output, image_ds,
                 gdal_frmt, ndv, band_names)

    image_ds = None


# UTILITY FUNCTIONS
def find_result_attributes(results, bands, coefs, prefix=''):
    """ Returns attributes about the dataset from result files

    Args:
        results (list): Result filenames
        bands (list): Bands to describe for output
        coefs (list): Coefficients to describe for output
        prefix (str, optional): Search for coef/rmse results with given prefix
            (default: '')

    Returns:
        tuple: Tuple containing `list` of indices for output bands and output
            coefficients, `bool` for outputting RMSE, `list` of coefficient
            names, `str` design specification, and `OrderedDict` design_info
            (i_bands, i_coefs, use_rmse, design, design_info)

    """
    _coef = prefix + 'coef' if prefix else 'coef'
    _rmse = prefix + 'rmse' if prefix else 'rmse'

    # How many coefficients and bands exist in the results?
    n_bands, n_coefs = None, None
    design = None
    for r in results:
        try:
            _result = np.load(r)
            rec = _result['record']
            design = _result['design_matrix'].item()
            design_str = _result['design'].item()
        except:
            continue

        if not rec.dtype.names:
            continue

        if _coef not in rec.dtype.names or _rmse not in rec.dtype.names:
            logger.error('Could not find coefficients ({0}) and RMSE ({1}) '
                         'in record'.format(_coef, _rmse))
            if prefix:
                logger.error('Coefficients and RMSE not found with prefix %s. '
                             'Did you calculate them?' % prefix)
            raise click.Abort()

        try:
            n_coefs, n_bands = rec[_coef][0].shape
        except:
            continue
        else:
            break

    if n_coefs is None or n_bands is None:
        logger.error('Could not determine the number of coefficients or bands')
        raise click.Abort()
    if design is None:
        logger.error('Design matrix specification not found in results.')
        raise click.Abort()

    # How many bands does the user want?
    if bands == 'all':
        i_bands = range(0, n_bands)
    else:
        # NumPy index on 0; GDAL on 1 -- so subtract 1
        i_bands = [b - 1 for b in bands]
        if any([b > n_bands for b in i_bands]):
            logger.error('Bands specified exceed size of bands in results')
            raise click.Abort()

    # How many coefficients did the user want?
    use_rmse = False
    if coefs:
        if 'rmse' in coefs or 'all' in coefs:
            use_rmse = True
        i_coefs, coef_names = design_to_indices(design, coefs)
    else:
        i_coefs, coef_names = None, None

    logger.debug('Bands: {0}'.format(i_bands))
    if coefs:
        logger.debug('Coefficients: {0}'.format(i_coefs))

    return (i_bands, i_coefs, use_rmse, coef_names, design_str, design)


def find_indices(record, date, after=False, before=False):
    """ Yield indices matching time segments for a given date

    Args:
      record (np.ndarray): Saved model result
      date (int): Ordinal date to use when finding matching segments
      after (bool, optional): If date intersects a disturbed period, use next
        available time segment
      before (bool, optional): If date does not intersect a model, use previous
        non-disturbed time segment

    Yields:
      tuple: (int, np.ndarray) the QA value and indices of `record` containing
        indices matching criteria. If `before` or `after` are specified,
        indices will be yielded in order of least desirability to allow
        overwriting -- `before` indices, `after` indices, and intersecting
        indices.

    """
    if before:
        # Model before, as long as it didn't change
        index = np.where((record['end'] <= date) & (record['break'] == 0))[0]
        yield _before_qa, index

    if after:
        # First model starting after date specified
        index = np.where(record['start'] >= date)[0]
        _, _index = np.unique(record['px'][index], return_index=True)
        index = index[_index]
        yield _after_qa, index

    # Model intersecting date
    index = np.where((record['start'] <= date) & (record['end'] >= date))[0]
    yield _intersect_qa, index


# MAPPING FUNCTIONS
def get_classification(date, result_location, image_ds,
                       after=False, before=False, qa=False,
                       pred_proba=False,
                       ndv=0, pattern=_result_record):
    """ Output raster with classification results

    Args:
        date (int): ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before)
        pred_proba (bool, optional): Include additional band with
            classification value probabilities
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results

    Returns:
        np.ndarray: 2D numpy array containing the classification map for the
            date specified

    """
    # Find results
    records = find_results(result_location, pattern)

    n_bands = 2 if pred_proba else 1
    dtype = np.uint16 if pred_proba else np.uint8

    band_names = ['Classification']
    if pred_proba:
        band_names.append('Pred Proba (x10,000)')
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=dtype) * int(ndv)

    logger.debug('Processing results')
    for rec, fname in iter_records(records, warn_on_empty=WARN_ON_EMPTY,
                                   yield_filename=True):
        if 'class' not in rec.dtype.names:
            logger.warning('Results in {f} do not have classification labels'
                           .format(f=fname))
            continue
        if 'class_proba' not in rec.dtype.names and pred_proba:
            raise ValueError('Results do not have classification prediction'
                             ' probability values')

        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            raster[rec['py'][index],
                   rec['px'][index], 0] = rec['class'][index]
            if pred_proba:
                raster[rec['py'][index],
                       rec['px'][index], 1] = \
                            rec['class_proba'][index].max(axis=1) * 10000
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names


def get_coefficients(date, result_location, image_ds,
                     bands, coefs,
                     prefix='', amplitude=False,
                     after=False, before=False, qa=False,
                     ndv=-9999, pattern=_result_record):
    """ Output a raster with coefficients from CCDC

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        bands (list): Bands to predict
        coefs (list): List of coefficients to output
        image_ds (gdal.Dataset): Example dataset
        prefix (str, optional): Use coef/rmse with refit prefix (default: '')
        amplitude (bool, optional): Map amplitude of seasonality instead of
            individual coefficient estimates for sin/cosine pair
            (default: False)
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment (default: False)
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment (default: False)
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before) (default: False)
        ndv (int, optional): NoDataValue (default: -9999)
        pattern (str, optional): filename pattern of saved record results

    Returns:
        tuple: A tuple (np.ndarray, list) containing the 3D numpy.ndarray of
            the coefficients (coefficient x band x pixel), and the band names
            for the output dataset

    """
    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, i_coefs, use_rmse, coef_names, _, _ = find_result_attributes(
        records, bands, coefs, prefix=prefix)

    # Process amplitude transform for seasonality coefficients
    if amplitude:
        harm_coefs = []
        for i, (c, n) in enumerate(zip(i_coefs, coef_names)):
            if re.match(r'harm\(x, [0-9]+\)\[0]', n):
                harm_coefs.append(c)
                coef_names[i] = re.sub(r'harm(.*)\[.*', r'amplitude\1', n)
        # Remove sin term from each harmonic pair
        i_coefs = [c for c in i_coefs if c - 1 not in harm_coefs]
        coef_names = [n for n in coef_names if 'harm' not in n]

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = n_bands if use_rmse else 0

    # Setup output band names
    band_names = []
    for _c in coef_names:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_' + _c.replace(' ', ''))
    if use_rmse is True:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_RMSE')
    n_qa = 0
    if qa:
        n_qa += 1
        band_names.append('SegmentQAQC')
    n_out_bands = n_bands * n_coefs + n_rmse + n_qa

    _coef = prefix + 'coef' if prefix else 'coef'
    _rmse = prefix + 'rmse' if prefix else 'rmse'

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_out_bands),
                     dtype=np.float32) * ndv

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            if n_coefs > 0:
                # Normalize intercept to mid-point in time segment
                rec[_coef][index, 0, :] += (
                    (rec['start'][index] + rec['end'][index])
                        / 2.0)[:, np.newaxis] * \
                    rec[_coef][index, 1, :]

                # If we want amplitude, calculate it
                if amplitude:
                    for harm_coef in harm_coefs:
                        rec[_coef][index, harm_coef, :] = np.linalg.norm(
                            rec[_coef][index, harm_coef:harm_coef + 2, :],
                            axis=1)

                # Extract coefficients
                raster[rec['py'][index],
                       rec['px'][index], :n_coefs * n_bands] =\
                    np.reshape(rec[_coef][index][:, i_coefs, :][:, :, i_bands],
                               (index.size, n_coefs * n_bands))

            if use_rmse:
                raster[rec['py'][index], rec['px'][index],
                       n_coefs * n_bands:n_out_bands - n_qa] =\
                    rec[_rmse][index][:, i_bands]
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names


def get_prediction(date, result_location, image_ds,
                   bands='all', prefix='',
                   after=False, before=False, qa=False,
                   ndv=-9999, pattern=_result_record):
    """ Output a raster with the predictions from model fit for a given date

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        bands (str, list): Bands to predict - 'all' for every band, or specify
            a list of bands
        prefix (str, optional): Use coef/rmse with refit prefix (default: '')
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before)
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results

    Returns:
        np.ndarray: A 3D numpy.ndarray containing the prediction for each band,
            for each pixel

    """
    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, _, _, _, design, design_info = find_result_attributes(
        records, bands, None, prefix=prefix)

    n_bands = len(i_bands)
    band_names = ['Band_{0}'.format(b) for b in range(n_bands)]
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')
    n_i_bands = len(i_bands)

    # Create X matrix from date -- ignoring categorical variables
    if re.match(r'.*C\(.*\).*', design):
        logger.warning('Categorical variable found in design matrix not used'
                       ' in predicted image estimate')
    design = re.sub(r'[\+\-][\ ]+C\(.*\)', '', design)
    X = patsy.dmatrix(design, {'x': date}).squeeze()

    i_coef = []
    for k, v in design_info.iteritems():
        if not re.match('C\(.*\)', k):
            i_coef.append(v)
    i_coef = np.asarray(i_coef)

    logger.debug('Allocating memory')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int16) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            # Calculate prediction
            _coef = rec['coef'].take(index, axis=0).\
                take(i_coef, axis=1).take(i_bands, axis=2)
            raster[rec['py'][index], rec['px'][index], :n_i_bands] = \
                np.tensordot(_coef, X, axes=(1, 0))
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names


def get_phenology(date, result_location, image_ds,
                  after=False, before=False, qa=False,
                  ndv=-9999, pattern=_result_record):
    """ Output a raster containing phenology information

    Phenology information includes spring_doy, autumn_doy, pheno_cor, peak_evi,
    peak_doy, and pheno_nobs.

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before)
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results

    Returns:
        tuple (np.ndarray, list): A tuple (np.ndarray, list) containing the 3D
            np.ndarray of the phenology metrics, and the band names for
            the output dataset

    """
    # Find results
    records = find_results(result_location, pattern)

    n_bands = 7
    attributes = ['spring_doy', 'autumn_doy', 'pheno_cor', 'peak_evi',
                  'peak_doy', 'pheno_nobs']
    band_names = ['SpringDOY', 'AutumnDOY', 'Pheno_R*10000', 'Peak_EVI*10000',
                  'Peak_DOY', 'Pheno_NObs', 'GrowingDOY']
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')

    logger.debug('Allocating memory')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
        if not all([_attr in rec.dtype.names for _attr in attributes]):
            raise ValueError('Results do not have phenology metrics')

        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            # Apply scale factors for R and peak EVI
            rec['pheno_cor'][index] *= 10000.0
            rec['peak_evi'][index] *= 10000.0

            for _b, _attr in enumerate(attributes):
                raster[rec['py'][index],
                       rec['px'][index], _b] = rec[_attr][index]
            raster[rec['py'][index],
                   rec['px'][index], 6] = \
                rec['autumn_doy'][index] - rec['spring_doy'][index]
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names
