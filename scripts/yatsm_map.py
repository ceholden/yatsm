#!/usr/bin/env python
""" Make map of YATSM output for a given date

Usage:
    yatsm_map.py [options] ( coef | predict | class | pheno ) <date> <output>

Options:
    --root <dir>            Root time series directory [default: ./]
    -r --result <dir>       Directory of results [default: YATSM]
    -i --image <image>      Example image [default: example_img]
    --ndv <NoDataValue>     No data value for map [default: 0]
    -f --format <format>    Output raster format [default: GTiff]
    --date <format>         Input date format [default: %Y-%m-%d]
    --warn-on-empty         Warn user when reading in empty result files
    --version               Show program version
    -v --verbose            Show verbose debugging messages
    -h --help               Show help messages

Time segment map options:
    --after                 Use time segment after <date> if needed for map
    --before                Use time segment before <date> if needed for map
    --qa                    Add QA band identifying segment type (3=intersect,
                                2=after, 1=before)

Classification map options:
    --predict_proba         Include prediction probability band (P x 10000)

Coefficient and prediction options:
    --band <bands>          Bands to export [default: all]
    --robust                Use robust estimates

Coefficient options:
    --no_scale_intercept    Don't scale intercept by slope term
    --coef <coefs>          Coefficients to export [default: all]

Examples:
    > yatsm_map.py --coef "intercept, slope" --band "3, 4, 5" --ndv -9999 coef
    ... 2000-01-01 coef_map.gtif

    > yatsm_map.py --date "%Y-%j" predict 2000-001 prediction.gtif

    > yatsm_map.py --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

Notes:
    - Image predictions will not use categorical information in timeseries
        models.

"""
from __future__ import division, print_function

import datetime as dt
import logging
import os
import re
import sys

from docopt import docopt
import numpy as np
from osgeo import gdal
import patsy

# Handle running as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.utils import find_results, iter_records, write_output
from yatsm.regression import design_to_indices, design_coefs
from yatsm.regression.transforms import harm

gdal.UseExceptions()
gdal.AllRegister()

FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

# QA/QC values for segment types
_intersect_qa = 3
_after_qa = 2
_before_qa = 1

# Filters for results
_result_record = 'yatsm_*'
# number of days in year
_days = 365.25
w = 2 * np.pi / _days

WARN_ON_EMPTY = False


# UTILITY FUNCTIONS
def find_result_attributes(results, bands, coefs, use_robust=False):
    """ Returns attributes about the dataset from result files

    Args:
      results (list): Result filenames
      bands (list): Bands to describe for output
      coefs (list): Coefficients to describe for output
      use_robust (bool, optional): Search for robust results

    Returns:
      tuple: Tuple containing `list` of indices for output bands and output
        coefficients, `bool` for outputting RMSE, `list` of coefficient names,
        `str` design specification, and `OrderedDict` design_info
        (i_bands, i_coefs, use_rmse, design, design_info)

    """
    _coef = 'robust_coef' if use_robust else 'coef'
    _rmse = 'robust_rmse' if use_robust else 'rmse'

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
            if use_robust:
                logger.error('Robust coefficients and RMSE not found. Did you '
                             'calculate them?')
            sys.exit(1)

        try:
            n_coefs, n_bands = rec[_coef][0].shape
        except:
            continue
        else:
            break

    if n_coefs is None or n_bands is None:
        logger.error('Could not determine the number of coefficients or bands')
        sys.exit(1)
    if design is None:
        logger.error('Design matrix specification not found in results.')
        sys.exit(1)

    # How many bands does the user want?
    if bands == 'all':
        i_bands = range(0, n_bands)
    else:
        # NumPy index on 0; GDAL on 1 -- so subtract 1
        i_bands = [b - 1 for b in bands]
        if any([b > n_bands for b in i_bands]):
            logger.error('Bands specified exceed size of bands in results')
            sys.exit(1)

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
      before (bool, optional): If date does not intersect a model, use previous
        non-disturbed time segment
      qa (bool, optional): Add QA flag specifying segment type (intersect,
        after, or before)
      pred_proba (bool, optional): Include additional band with classification
        value probabilities
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      np.ndarray: 2D numpy array containing the classification map for the date
        specified

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
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
        if 'class' not in rec.dtype.names:
            raise ValueError('Results do not have classification labels')
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
                     no_scale_intercept=False, use_robust=False,
                     after=False, before=False, qa=False,
                     ndv=-9999, pattern=_result_record):
    """ Output a raster with coefficients from CCDC

    Args:
      date (int): Ordinal date for prediction image
      result_location (str): Location of the results
      bands (list): Bands to predict
      coefs (list): List of coefficients to output
      image_ds (gdal.Dataset): Example dataset
      no_scale_intercept (bool, optional): Skip scaling of intercept
        coefficient by slope (default: False)
      use_robust (bool, optional): Map robust coefficients and RMSE instead of
        normal ones
      after (bool, optional): If date intersects a disturbed period, use next
        available time segment
      before (bool, optional): If date does not intersect a model, use previous
        non-disturbed time segment
      qa (bool, optional): Add QA flag specifying segment type (intersect,
        after, or before)
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      tuple: A tuple (np.ndarray, list) containing the 3D numpy.ndarray of the
        coefficients (coefficient x band x pixel), and the band names for
        the output dataset

    """
    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, i_coefs, use_rmse, coef_names, _, _ = find_result_attributes(
        records, bands, coefs, use_robust=use_robust)

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

    logger.debug('Band names:')
    logger.debug(band_names)

    _coef = 'robust_coef' if use_robust else 'coef'
    _rmse = 'robust_rmse' if use_robust else 'rmse'

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize,
                     n_out_bands),
                     dtype=np.float32) * ndv

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=WARN_ON_EMPTY):
        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            if n_coefs > 0:
                # Normalize intercept to mid-point in time segment
                if not no_scale_intercept:
                    rec[_coef][index, 0, :] += (
                        (rec['start'][index] + rec['end'][index])
                        / 2.0)[:, None] * rec[_coef][index, 1, :]

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
                   bands='all', use_robust=False,
                   after=False, before=False, qa=False,
                   ndv=-9999, pattern=_result_record):
    """ Output a raster with the predictions from model fit for a given date

    Args:
      date (int): Ordinal date for prediction image
      result_location (str): Location of the results
      image_ds (gdal.Dataset): Example dataset
      bands (str, list): Bands to predict - 'all' for every band, or specify a
        list of bands
      use_robust (bool, optional): Map robust coefficients and RMSE instead of
        normal ones
      after (bool, optional): If date intersects a disturbed period, use next
        available time segment
      before (bool, optional): If date does not intersect a model, use previous
        non-disturbed time segment
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
        records, bands, None, use_robust=use_robust)

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
      before (bool, optional): If date does not intersect a model, use previous
        non-disturbed time segment
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


def parse_args(args):
    """ Returns dictionary of parsed and validated command arguments

    Args:
      args (dict): Arguments from user

    Returns:
      dict: Parsed and validated arguments

    """
    parsed = {}
    ### Parse required input
    # Type of map
    parsed['class'] = args['class']
    parsed['coef'] = args['coef']
    parsed['predict'] = args['predict']
    parsed['pheno'] = args['pheno']

    # Date for map
    date = args['<date>']
    date_format = args['--date']
    try:
        date = dt.datetime.strptime(date, date_format)
    except:
        logger.error('Could not parse date')
        raise
    parsed['date'] = date.toordinal()

    # Output name
    parsed['output'] = os.path.abspath(args['<output>'])
    if not os.path.isdir(os.path.dirname(parsed['output'])):
        try:
            os.makedirs(os.path.dirname(parsed['output']))
        except:
            logger.error('Could not make output directory specified')
            raise

    ### Parse generic options
    # NDV
    try:
        parsed['ndv'] = float(args['--ndv'])
    except ValueError:
        logger.error('NoDataValue must be a real number')
        raise

    # Root directory
    parsed['root'] = args['--root']
    if not os.path.isdir(parsed['root']):
        logger.error('Root directory is not a directory')
        sys.exit(1)

    # Results folder
    results = args['--result']
    # First look as relative path under root folder, then from PWD
    if not os.path.isdir(os.path.join(parsed['root'], results)):
        if os.path.isdir(results):
            results = os.path.abspath(results)
        else:
            logger.error('Cannot find results folder')
            sys.exit(1)
    else:
        results = os.path.abspath(os.path.join(parsed['root'], results))
    parsed['results'] = results

    # Example image
    image = args['--image']
    if not os.path.isfile(image):
        if os.path.isfile(os.path.join(parsed['root'], image)):
            image = os.path.join(parsed['root'], image)
        else:
            logger.error('Cannot find example image')
            sys.exit(1)
    parsed['image'] = os.path.abspath(image)

    # Raster file format
    parsed['gdal_frmt'] = args['--format']
    try:
        _ = gdal.GetDriverByName(parsed['gdal_frmt'])
    except:
        logger.error('Unknown GDAL format specified')
        raise

    ### Parse coefficient and prediction options
    # Coefficients to output
    parsed['coefs'] = [c for c in
                       args['--coef'].replace(',', ' ').split(' ') if c != '']
    if not all([c.lower() in design_coefs for c in parsed['coefs']]):
        logger.error('Unknown coefficient options')
        logger.error('Options are:')
        logger.error(design_coefs)
        logger.error('Specified were:')
        logger.error(parsed['coefs'])
        sys.exit(1)

    # Intercept coefficient handling
    parsed['no_scale_intercept'] = args['--no_scale_intercept']

    # Bands to output
    bands = args['--band']
    if bands != 'all':
        bands = bands.replace(',', ' ').split(' ')
        try:
            bands = [int(b) for b in bands if b != '']
        except ValueError:
            logger.error('Band specification must be "all" or integers')
            raise
        except:
            logger.error('Could not parse band selection')
            raise
    parsed['bands'] = bands

    # Robust estimates?
    parsed['use_robust'] = args['--robust']

    ### Classification outputs
    parsed['pred_proba'] = args['--predict_proba']

    ### Generic map options
    # Go to next time segment option
    parsed['after'] = args['--after']
    parsed['before'] = args['--before']
    parsed['qa'] = args['--qa']

    return parsed


def main(args):
    """ Test input and pass to appropriate functions """
    ### Produce output specified
    try:
        image_ds = gdal.Open(args['image'], gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    band_names = None
    if args['class']:
        raster, band_names = get_classification(
            args['date'], args['results'], image_ds,
            after=args['after'], before=args['before'], qa=args['qa'],
            pred_proba=args['pred_proba']
        )
    elif args['coef']:
        raster, band_names = get_coefficients(
            args['date'], args['results'], image_ds,
            args['bands'], args['coefs'],
            no_scale_intercept=args['no_scale_intercept'],
            use_robust=args['use_robust'],
            after=args['after'], before=args['before'], qa=args['qa'],
            ndv=args['ndv']
        )
    elif args['predict']:
        raster, band_names = get_prediction(
            args['date'], args['results'], image_ds,
            args['bands'],
            use_robust=args['use_robust'],
            after=args['after'], before=args['before'], qa=args['qa'],
            ndv=args['ndv']
        )
    elif args['pheno']:
        raster, band_names = get_phenology(
            args['date'], args['results'], image_ds,
            after=args['after'], before=args['before'], qa=args['qa'],
            ndv=args['ndv'])

    write_output(raster, args['output'], image_ds,
                 args['gdal_frmt'], args['ndv'], band_names)

    image_ds = None


if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--warn-on-empty']:
        WARN_ON_EMPTY = True

    args = parse_args(args)
    main(args)
