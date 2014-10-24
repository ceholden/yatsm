#!/usr/bin/env python
""" Make map of YATSM output for a given date

Usage:
    yatsm_map.py [options] ( coef | predict | class ) <date> <output>

Options:
    --ndv <NoDataValue>     No data value for map [default: 0]
    --root <dir>            Root time series directory [default: ./]
    -r --result <dir>       Directory of results [default: YATSM]
    -i --image <image>      Example image [default: example_img]
    --date <format>         Date format [default: %Y-%m-%d]
    -f --format <format>    Output raster format [default: GTiff]
    --warn-on-empty         Warn user when reading in empty result files
    -v --verbose            Show verbose debugging messages
    -h --help               Show help messages

Coefficient options:
    --coef <coefs>          Coefficients to export [default: all]
    --band <bands>          Bands to export [default: all]

Prediction options:

Class options:
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
logger = logging.getLogger(__name__)

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


def get_classification(date, after, before, results, image_ds,
                       ndv=0, pattern=_result_record):
    """ Output raster with classification results

    Args:
      date (int): ordinal date for prediction image
      after (bool): If date intersects a disturbed period, use next segment?
      before (bool): If date intersects a disturbed period, use previous
        segment?
      results (str): Location of the results
      image_ds (gdal.Dataset): Example dataset
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      raster (np.array): 2D numpy array containing the classification map for
        the date specified

    """
    # Init output raster
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.uint8) * ndv

    records = find_results(results, pattern)
    n_records = len(records)

    for i, r in enumerate(records):
        if np.mod(i, 100) == 0:
            logger.debug('{0:.0f}%'.format(i / n_records * 100))

        rec = np.load(r)['record']

        if not 'class' in rec.dtype.names:
            raise ValueError('Record {r} does not have classification \
                labels'.format(r=r))

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


def get_coefficients(date, bands, coefs, results, image_ds,
                     ndv=-9999, pattern=_result_record):
    """ Output a raster with coefficients from CCDC

    Args:
      date (int): Ordinal date for prediction image
      bands (list): Bands to predict
      coefs (list): List of coefficients to output
      results (str): Location of the results
      image_ds (gdal.Dataset): Example dataset
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
        (raster, band_names):   A tuple containing the 3D numpy.ndarray
                                containing the coefficients for each band, for
                                each pixel, and the band names for the output
                                dataset

    """
    # Find results
    records = find_results(results, pattern)
    n_records = len(records)

    # Find how many coefficients there are for output
    n_coef = None
    n_band = None
    for i, r in enumerate(records):
        try:
            rec = np.load(r)['record']
        except:
            continue

        try:
            n_coef, n_band = rec['coef'][0].shape
        except:
            continue
        else:
            break

    if not n_coef or not n_band:
        logger.error('Could not determine the number of coefficients')
        sys.exit(1)

    # Find how many bands are used in output
    i_bands = []
    if bands == 'all':
        i_bands = range(0, n_band)
    else:
        # numpy index on 0; GDAL index on 1 so subtract 1
        i_bands = [b - 1 for b in bands]
        if any([b > n_band for b in i_bands]):
            logger.error('Bands specified exceed size of coefficients \
                         in results')
            sys.exit(1)

    # Determine indices for the coefficients desired
    i_coefs = []
    use_rmse = False
    for c in coefs:
        if c == 'all':
            i_coefs.extend(range(0, n_coef))
            use_rmse = True
            break
        elif c == 'intercept':
            i_coefs.append(0)
        elif c == 'slope':
            i_coefs.append(1)
        elif c == 'seasonality':
            i_coefs.extend(range(2, n_coef))
        elif c == 'rmse':
            use_rmse = True

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = 0
    if use_rmse is True:
        n_rmse = n_bands

    logger.debug('Indices for bands and coefficients:')
    logger.debug('Bands:')
    logger.debug(i_bands)
    logger.debug('Coefficients:')
    logger.debug(i_coefs)

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

    logger.debug('Processing results')

    for _i, r in enumerate(records):
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.0f}%'.format(_i / n_records * 100))

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

        # Find indices for the date specified
        index = np.where((rec['start'] <= date) & (rec['end'] >= date))[0]

        if index.shape[0] == 0:
            continue

        # Normalize intercept to mid-point in time segment
        rec['coef'][index, 0, :] += \
            ((rec['start'][index] + rec['end'][index]) / 2.0)[:, None] * \
            rec['coef'][index, 1, :]

        # Extract coefficients
        raster[rec['py'][index], rec['px'][index], :n_coefs * n_bands] =\
            np.reshape(rec['coef'][index][:, i_coefs, :][:, :, i_bands],
                       (index.size, n_coefs * n_bands))

        if use_rmse:
            raster[rec['py'][i], rec['px'][i], n_coefs * n_bands:] = \
                rec['rmse'][i][i_bands]

    return (raster, band_names)


def get_prediction(date, bands, results, image_ds,
                   ndv=-9999, pattern=_result_record):
    """ Output a raster with the predictions from model fit for a given date

    Args:
      date (int): Ordinal date for prediction image
      coefs (list): List of coefficients to output
      results (str): Location of the results
      image_ds (gdal.Dataset): Example dataset
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
        A 3D numpy.ndarray containing the prediction for each band, for each
        pixel

    """
    # raise NotImplementedError("Haven't ported this yet...")
    # Find results
    records = find_results(results, pattern)
    n_records = len(records)

    # Find how many coefficients there are for output
    n_coef = None
    n_band = None
    freq = None
    for i, r in enumerate(records):
        try:
            _result = np.load(r)
            rec = _result['record']
            freq = _result['freq']
        except:
            continue

        try:
            n_coef, n_band = rec['coef'][0].shape
        except:
            continue
        else:
            break

    if not n_coef or not n_band:
        logger.error('Could not determine the number of coefficients')
        sys.exit(1)
    if not freq:
        logger.error('Seasonality frequency not found in results.')
        sys.exit(1)

    # Create X matrix from date
    X = make_X(date, freq)

    # Find how many bands are used in output
    i_bands = []
    if bands == 'all':
        i_bands = range(0, n_band)
    else:
        # numpy index on 0; GDAL index on 1 so subtract 1
        i_bands = [b - 1 for b in bands]
        if any([b > n_band for b in i_bands]):
            logger.error('Bands specified exceed size of coefficients \
                         in results')
            sys.exit(1)

    n_band = len(i_bands)

    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_band),
                     dtype=np.int16) * ndv

    for _i, r in enumerate(records):
        # Verbose progress
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.0f}%'.format(_i / n_records * 100))
        # Open output
        try:
            rec = np.load(r)['record']
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(f=r))
            continue

        if rec.shape[0] == 0:
            # No values in this file
            logger.warning('Could not find results in {f}'.format(f=r))
            continue

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


def main():
    """ Test input and pass to appropriate functions """
    ### Parse required input
    # Date for map
    date = args['<date>']
    date_format = args['--date']
    try:
        date = dt.datetime.strptime(date, date_format)
    except:
        logger.error('Could not parse date')
        raise
    date = date.toordinal()

    # Output name
    output = os.path.abspath(args['<output>'])
    if not os.path.isdir(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except:
            logger.error('Could not make output directory specified')
            raise

    ### Parse generic options
    # NDV
    try:
        ndv = float(args['--ndv'])
    except ValueError:
        logger.error('NoDataValue must be a real number')
        raise

    # Root directory
    root = args['--root']
    if not os.path.isdir(root):
        logger.error('Root directory is not a directory')
        sys.exit(1)

    # Results folder
    results = args['--result']
    if not os.path.isdir(results):
        if os.path.isdir(os.path.join(root, results)):
            results = os.path.join(root, results)
        else:
            logger.error('Cannot find results folder')
            sys.exit(1)
    results = os.path.abspath(results)

    # Example image
    image = args['--image']
    if not os.path.isfile(image):
        if os.path.isfile(os.path.join(root, image)):
            image = os.path.join(root, image)
        else:
            logger.error('Cannot find example image')
            sys.exit(1)
    image = os.path.abspath(image)

    # Raster file format
    gdal_frmt = args['--format']
    try:
        _ = gdal.GetDriverByName(gdal_frmt)
    except:
        logger.error('Unknown GDAL format specified')
        raise

    ### Parse coefficient options
    # Coefficients to output
    coefs = [c for c in args['--coef'].replace(',', ' ').split(' ') if c != '']
    if not all([c.lower() in _coefs for c in coefs]):
        logger.error('Unknown coefficient options')
        logger.error('Options are:')
        logger.error(_coefs)
        logger.error('Specified were:')
        logger.error(coefs)
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

    ### Parse prediction options

    ### Classification outputs
    # Go to next time segment option
    after = args['--after']
    before = args['--before']

    ### Produce output specified
    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    band_names = None

    if args['class']:
        raster = get_classification(date, after, before, results, image_ds)
    elif args['coef']:
        raster, band_names = \
            get_coefficients(date, bands, coefs, results, image_ds, ndv=ndv)
    elif args['predict']:
        raster = get_prediction(date, bands, results, image_ds, ndv=ndv)

    if args['class']:
        write_output(raster, output, image_ds, gdal_frmt, ndv, band_names)
    else:
        write_output(raster, output, image_ds, gdal_frmt, ndv, band_names)

    image_ds = None


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--warn-on-empty']:
        WARN_ON_EMPTY = True

    main()
