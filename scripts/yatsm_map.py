#!/usr/bin/env python
""" Make map of YATSM output for a given date

Usage:
    yatsm_map.py [options] ( coef | predict | class ) <date> <output>

Options:
    --root <dir>            Root time series directory [default: ./]
    -r --result <dir>       Directory of results [default: YATSM]
    -i --image <image>      Example image [default: example_img]
    --ndv <NoDataValue>     No data value for map [default: 0]
    -f --format <format>    Output raster format [default: GTiff]
    --date <format>         Date format [default: %Y-%m-%d]
    --warn-on-empty         Warn user when reading in empty result files
    -v --verbose            Show verbose debugging messages
    -h --help               Show help messages

Coefficient and prediction options:
    --band <bands>          Bands to export [default: all]
    --robust                Use robust estimates
    --coef <coefs>          Coefficients to export [default: all]

Classification options:
    --after                 Use time segment after <date> if needed for map
    --before                Use time segment before <date> if needed for map

Examples:
    > yatsm_map.py --coef "intercept, slope" --band "3, 4, 5" --ndv -9999 coef
    ... 2000-01-01 coef_map.gtif

    > yatsm_map.py --date "%Y-%j" predict 2000-001 prediction.gtif

    > yatsm_map.py --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

"""
from __future__ import division, print_function

import datetime as dt
import fnmatch
import logging
import os
import sys

from docopt import docopt
import numpy as np
from osgeo import gdal, gdal_array

gdal.UseExceptions()
gdal.AllRegister()

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

# Possible coefficients
_coefs = ['all', 'intercept', 'slope', 'seasonality', 'rmse']
# Filters for results
_result_record = 'yatsm_*'
# number of days in year
_days = 365.25
w = 2 * np.pi / _days

WARN_ON_EMPTY = False


# UTILITY FUNCTIONS
def find_results(location, pattern):
    """ Create list of result files and return sorted

    Args:
      location (str): directory location to search
      pattern (str): glob style search pattern for results

    Returns:
      results (list): list of file paths for results found

    """
    # Note: already checked for location existence in main()
    records = []
    for root, dirnames, filenames in os.walk(location):
        for filename in fnmatch.filter(filenames, pattern):
            records.append(os.path.join(root, filename))

    if len(records) == 0:
        logger.error('Error: could not find results in: {0}'.format(location))
        sys.exit(1)

    records.sort()

    return records


def find_result_attributes(results, output_bands, output_coefs,
                           use_robust=False):
    """ Returns attributes about the dataset from result files

    Args:
      results (list): Result filenames
      output_bands (list): Bands to describe for output
      output_coefs (list): Coefficients to describe for output
      use_robust (bool, optional): Search for robust results

    Returns:
      tuple: Tuple containing list of indices for output bands and output
        coefficients, bool for outputting RMSE, and list of frequency of
        seasonality used in fit (i_bands, i_coefs, use_rmse, freq)

    """
    _coef = 'robust_coef' if use_robust else 'coef'
    _rmse = 'robust_rmse' if use_robust else 'rmse'

    # How many coefficients and bands exist in the results?
    result_bands = None
    result_coefs = None
    freq = None
    for r in results:
        try:
            _result = np.load(r)
            rec = _result['record']
            freq = _result['freq']
        except:
            continue

        if _coef not in rec.dtype.names or _rmse not in rec.dtype.names:
            logger.error('Could not find coefficients ({0}) and RMSE ({1}) '
                         'in record'.format(_coef, _rmse))
            if use_robust:
                logger.error('Robust coefficients and RMSE not found. Did you '
                             'calculate them?')
            sys.exit(1)

        try:
            result_coefs, result_bands = rec[_coef][0].shape
        except:
            continue
        else:
            break

    if not result_bands or not result_coefs:
        logger.error('Could not determine the number of coefficients or bands')
        sys.exit(1)
    if not freq:
        logger.error('Seasonality frequency not found in results.')
        sys.exit(1)

    # How many bands does the user want?
    if output_bands == 'all':
        i_bands = range(0, result_bands)
    else:
        # NumPy index on 0; GDAL on 1 -- so subtract 1
        i_bands = [b - 1 for b in output_bands]
        if any([b > result_bands for b in i_bands]):
            logger.error('Bands specified exceed size of bands in results')
            sys.exit(1)

    # How many coefficients did the user want?
    use_rmse = False
    i_coefs = []
    if output_coefs:
        for c in output_coefs:
            if c == 'all':
                i_coefs.extend(range(0, result_coefs))
                use_rmse = True
                break
            elif c == 'intercept':
                i_coefs.append(0)
            elif c == 'slope':
                i_coefs.append(1)
            elif c == 'seasonality':
                i_coefs.extend(range(2, result_coefs))
            elif c == 'rmse':
                use_rmse = True

    logger.debug('Bands: {0}'.format(i_bands))
    if output_coefs:
        logger.debug('Coefficients: {0}'.format(i_coefs))

    return (i_bands, i_coefs, use_rmse, freq)


def iter_records(records):
    """ Iterates over records, returning result NumPy array

    Args:
      records (list): List containing filenames of results

    Yields:
      np.ndarray: Result saved in record

    """
    n_records = len(records)

    for _i, r in enumerate(records):
        # Verbose progress
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.1f}%'.format(_i / n_records * 100))
        # Open output
        try:
            rec = np.load(r)['record']
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(f=r))
            continue

        if rec.shape[0] == 0:
            # No values in this file
            if WARN_ON_EMPTY:
                logger.warning('Could not find results in {f}'.format(f=r))
            continue

        yield rec


def make_X(x, freq, intercept=True):
    """ Create X matrix of Fourier series style independent variables

    Args:
        x               base of independent variables - dates
        freq            frequency of cosine/sin waves
        intercept       include intercept in X matrix

    Output:
        X               matrix X of independent variables

    Example:
        call:
            make_X(np.array([1, 2, 3]), [1, 2])
        returns:
            array([[ 1.        ,  1.        ,  1.        ],
                   [ 1.        ,  2.        ,  3.        ],
                   [ 0.99985204,  0.99940821,  0.99866864],
                   [ 0.01720158,  0.03439806,  0.05158437],
                   [ 0.99940821,  0.99763355,  0.99467811],
                   [ 0.03439806,  0.06875541,  0.10303138]])

    """
    if isinstance(x, int) or isinstance(x, float):
        x = np.array(x)

    if intercept:
        X = np.array([np.ones_like(x), x])
    else:
        X = x

    for f in freq:
        if X.ndim == 2:
            X = np.vstack([X, np.array([
                np.cos(f * w * x),
                np.sin(f * w * x)])
            ])
        elif X.ndim == 1:
            X = np.concatenate((X,
                                np.array([
                                         np.cos(f * w * x),
                                         np.sin(f * w * x)])
                                ))

    return X


def get_classification(date, result_location, image_ds,
                       after=False, before=False,
                       ndv=0, pattern=_result_record):
    """ Output raster with classification results

    Args:
      date (int): ordinal date for prediction image
      result_location (str): Location of the results
      image_ds (gdal.Dataset): Example dataset
      after (bool, optional): If date intersects a disturbed period, use next
        segment?
      before (bool, optional): If date intersects a disturbed period, use
        previous segment?
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      np.ndarray: 2D numpy array containing the classification map for the date
        specified

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.uint8) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records):
        if not 'class' in rec.dtype.names:
            raise ValueError('Results do not have classification labels')

        # Find model before segment
        if before:
            # Model before, as long as it didn't change
            index = np.where((rec['end'] <= date) & (rec['break'] == 0))[0]
            if index.shape[0] != 0:
                raster[rec['py'][index],
                       rec['px'][index]] = rec['class'][index]
        # Find model after segment
        if after:
            index = np.where(rec['start'] >= date)[0]
            _, _index = np.unique(rec['px'][index], return_index=True)
            index = index[_index]
            if index.shape[0] != 0:
                raster[rec['py'][index],
                       rec['px'][index]] = rec['class'][index]

        # Find model intersecting date
        index = np.where((rec['start'] <= date) & (rec['end'] >= date))[0]
        if index.shape[0] != 0:
            raster[rec['py'][index],
                   rec['px'][index]] = rec['class'][index]

    return raster


def get_coefficients(date, result_location, image_ds,
                     bands, coefs,
                     use_robust=False,
                     ndv=-9999, pattern=_result_record):
    """ Output a raster with coefficients from CCDC

    Args:
      date (int): Ordinal date for prediction image
      result_location (str): Location of the results
      bands (list): Bands to predict
      coefs (list): List of coefficients to output
      image_ds (gdal.Dataset): Example dataset
      use_robust (bool, optional): Map robust coefficients and RMSE instead of
        normal ones
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
    i_bands, i_coefs, use_rmse, _ = find_result_attributes(
        records, bands, coefs, use_robust=use_robust)

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = n_bands if use_rmse else 0

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize,
                     n_bands * n_coefs + n_rmse),
                     dtype=np.float32) * ndv

    # Setup output band names
    band_names = []
    for _c in i_coefs:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_beta' + str(_c))
    for _b in i_bands:
        if use_rmse is True:
            band_names.append('B' + str(_b + 1) + '_RMSE')

    logger.debug('Band names:')
    logger.debug(band_names)

    _coef = 'robust_coef' if use_robust else 'coef'
    _rmse = 'robust_rmse' if use_robust else 'rmse'

    logger.debug('Processing results')
    for rec in iter_records(records):
        # Find indices for the date specified
        index = np.where((rec['start'] <= date) & (rec['end'] >= date))[0]

        if index.shape[0] == 0:
            continue

        # Normalize intercept to mid-point in time segment
        rec[_coef][index, 0, :] += \
            ((rec['start'][index] + rec['end'][index]) / 2.0)[:, None] * \
            rec[_coef][index, 1, :]

        # Extract coefficients
        raster[rec['py'][index], rec['px'][index], :n_coefs * n_bands] =\
            np.reshape(rec[_coef][index][:, i_coefs, :][:, :, i_bands],
                       (index.size, n_coefs * n_bands))

        if use_rmse:
            raster[rec['py'][index], rec['px'][index], n_coefs * n_bands:] =\
                rec[_rmse][index][:, i_bands]

    return (raster, band_names)


def get_prediction(date, result_location, image_ds,
                   bands='all', use_robust=False,
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
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
        A 3D numpy.ndarray containing the prediction for each band, for each
        pixel

    """
    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, _, _, freq = find_result_attributes(
        records, bands, None, use_robust=use_robust)

    n_bands = len(i_bands)

    # Create X matrix from date
    X = make_X(date, freq)

    logger.debug('Allocating memory')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int16) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records):
        # Find indices for the date specified
        index = np.where((rec['start'] <= date) & (rec['end'] >= date))[0]

        if index.shape[0] == 0:
            continue

        # Calculate prediction
        raster[rec['py'][index], rec['px'][index], :] = \
            np.tensordot(rec['coef'][index, :][:, :, i_bands], X, axes=(1, 0))

    return raster


def write_output(raster, output, image_ds, gdal_frmt, ndv, band_names=None):
    """ Write raster to output file """
    logger.debug('Writing output to disk')

    driver = gdal.GetDriverByName(gdal_frmt)

    if len(raster.shape) > 2:
        nband = raster.shape[2]
    else:
        nband = 1

    ds = driver.Create(
        output,
        image_ds.RasterXSize, image_ds.RasterYSize, nband,
        gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype.type)
    )

    if band_names is not None:
        if len(band_names) != nband:
            logger.error('Did not get enough names for all bands')
            sys.exit(1)

    if raster.ndim > 2:
        for b in range(nband):
            logger.debug('    writing band {b}'.format(b=b + 1))
            ds.GetRasterBand(b + 1).WriteArray(raster[:, :, b])
            ds.GetRasterBand(b + 1).SetNoDataValue(ndv)

            if band_names is not None:
                ds.GetRasterBand(b + 1).SetDescription(band_names[b])
    else:
        logger.debug('    writing band')
        ds.GetRasterBand(1).WriteArray(raster)
        ds.GetRasterBand(1).SetNoDataValue(ndv)

        if band_names is not None:
            ds.GetRasterBand(1).SetDescription(band_names[0])

    ds.SetProjection(image_ds.GetProjection())
    ds.SetGeoTransform(image_ds.GetGeoTransform())

    ds = None


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
    if not all([c.lower() in _coefs for c in parsed['coefs']]):
        logger.error('Unknown coefficient options')
        logger.error('Options are:')
        logger.error(_coefs)
        logger.error('Specified were:')
        logger.error(parsed['coefs'])
        sys.exit(1)

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
    # Go to next time segment option
    parsed['after'] = args['--after']
    parsed['before'] = args['--before']

    return parsed


def main(args):
    """ Test input and pass to appropriate functions """
    args = parse_args(args)
    ### Produce output specified
    try:
        image_ds = gdal.Open(args['image'], gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    band_names = None
    if args['class']:
        raster = get_classification(
            args['date'], args['results'], image_ds,
            args['after'], args['before']
        )
    elif args['coef']:
        raster, band_names = get_coefficients(
            args['date'], args['results'], image_ds,
            args['bands'], args['coefs'],
            use_robust=args['use_robust'],
            ndv=args['ndv']
        )
    elif args['predict']:
        raster = get_prediction(
            args['date'], args['results'], image_ds,
            args['bands'],
            use_robust=args['use_robust'],
            ndv=args['ndv']
        )

    write_output(raster, args['output'], image_ds,
                 args['gdal_frmt'], args['ndv'], band_names)

    image_ds = None


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--warn-on-empty']:
        WARN_ON_EMPTY = True

    main(args)
