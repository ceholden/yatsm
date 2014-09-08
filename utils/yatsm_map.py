#!/usr/bin/env python
""" Make map of YATSM output for a given date

Usage:
    yatsm_map.py [options] ( coef | predict | class ) <date> <output>

Option:
    --band <bands>          Bands to export [default: all]
    --coef <coefs>          Coefficients to export [default: all]
    --after                 Use time segment after <date> if needed
    --before                Use time segment before <date> if needed
    --ndv <NoDataValue>     No data value for classifications [default: 0]
    -d --directory <dir>    Root time series directory [default: ./]
    -r --result <dir>       Directory of results [default: YATSM]
    -i --image <image>      Example image [default: example_img]
    --date <format>         Date format [default: %Y-%m-%d]
    -f --format <format>    Output raster format [default: GTiff]
    -v --verbose            Show verbose debugging messages
    -h --help               Show help messages

Examples:
    yatsm_map.py --coef "intercept, slope" --band "3, 4, 5" --ndv -9999 coef
        2000-01-01 coef_map.gtif

    yatsm_map.py --after --date "%Y-%j" predict 2000-001 prediction.gtif

    yatsm_map.py --result "YATSM_new" --after class 2000-01-01 LCmap.gtif

"""
from __future__ import division, print_function

import datetime as dt
import fnmatch
import itertools
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
# Filters for CCDC results
_result_record = 'yatsm_*'
# number of days in year
_days = 365.25
w = 2 * np.pi / _days


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

    return records.sort()


def get_classification(date, after, before, results, image_ds,
                       pattern=_result_record):
    """ Output raster with classification results

    Args:
      date (int): ordinal date for prediction image
      after (bool): If date intersects a disturbed period, use next segment?
      before (bool): If date intersects a disturbed period, use previous
        segment?
      results (str): Location of the results
      image_ds (gdal.Dataset): Example dataset
      pattern (str, optional): filename pattern of saved record results

    Returns:
      raster (np.array): 2D numpy array containing the classification map for
        the date specified

    """
    # Init output raster
    raster = np.zeros((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.uint8)

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
            index = np.where(rec['end'] <= date)[0]
            if index.shape[0] != 0:
                raster[rec['py'][index],
                       rec['px'][index]] = rec['class'][index]
        # Find model after segment
        if after:
            index = np.where(rec['start'] >= date)[0]
            _, _index = np.unique(rec['px'], return_index=True)
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


def get_coefficients(date, after, bands, coefs, results, image_ds):
    """ Output a raster with coefficients from CCDC

    Args:
        date (int):     MATLAB datenum for prediction image
        after (bool):   If date intersects a disturbed period, use next segment
        bands (list):   Bands to predict
        coefs (list):   List of coefficients to output
        results (str):  Location of the CCDC results
        image_ds (gdal.Dataset):    Example dataset

    Returns:
        (raster, band_names):   A tuple containing the 3D numpy.ndarray
                                containing the coefficients for each band, for
                                each pixel, and the band names for the output
                                dataset

    """
    raise NotImplementedError("Haven't ported this yet...")
    # Find results
    mats = find_results(results, _rec_cg)
    n_mat = len(mats)

    # Find how many coefficients there are for output
    n_coef = None
    n_band = None
    for i, m in enumerate(mats):
        try:
            mat = spio.loadmat(m, squeeze_me=True,
                               struct_as_record=True)['rec_cg']
        except:
            continue

        try:
            n_coef, n_band = mat['coefs'][0].shape
        except:
            continue
        else:
            break

    if n_coef is None or n_band is None:
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
    logger.debug(i_bands)
    logger.debug(i_coefs)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize,
                     n_bands * n_coefs + n_rmse),
                     dtype=np.float32) * -9999

    # Setup output band names
    band_names = []
    for _b in i_bands:
        for _c in i_coefs:
            band_names.append('B' + str(_b + 1) + '_beta' + str(_c))
        if use_rmse is True:
            band_names.append('B' + str(_b + 1) + '_RMSE')

    logger.debug('Processing results')

    for _i, m in enumerate(mats):
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.0f}%'.format(_i / n_mat * 100))

        # Open MATLAB output
        try:
            mat = spio.loadmat(m, squeeze_me=True,
                               struct_as_record=True)['rec_cg']
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(f=m))
            continue

        if mat.ndim == 0:
            # No values in this .mat file
            continue

        # Find indices for the date specified
        if after is True:
            index = np.where(mat['t_end'] >= date)[0]
            _, _index = np.unique(mat['pos'][index], return_index=True)
            index = index[_index]
        else:
            index = np.where((mat['t_start'] <= date) &
                             (mat['t_end'] >= date))[0]

        if index.shape[0] == 0:
            continue

        # Locate entries in the map
        [rows, cols] = np.unravel_index(
            mat['pos'][index].astype(int) - 1,
            (image_ds.RasterYSize, image_ds.RasterXSize),
            order='C')

        for i, r, c in itertools.izip(index, rows, cols):
            # Normalize intercept to mid-point in time segment
            mat['coefs'][i][0, :] = mat['coefs'][i][0, :] + \
                (mat['t_start'][i] + mat['t_end'][i]) / 2.0 * \
                mat['coefs'][i][1, :]

            # Extract coefficients
            raster[r, c, range(n_coefs * n_bands)] = \
                mat['coefs'][i][i_coefs, :][:, i_bands].flatten()

            # Extract RMSE
            if use_rmse is True:
                raster[r, c, n_coefs * n_bands:] = mat['rmse'][i][i_bands]

    return (raster, band_names)


def get_prediction(date, after, bands, results, image_ds):
    """ Output a raster with the predictions from model fit for a given date

    Args:
        date (int):     MATLAB datenum for prediction image
        after (bool):   If date intersects a disturbed period, use next segment
        bands (list):   Bands to predict
        results (str):  Location of the CCDC results
        image_ds (gdal.Dataset):    Example dataset

    Returns:
        A 3D numpy.ndarray containing the prediction for each band, for each
        pixel
    """
    raise NotImplementedError("Haven't ported this yet...")
    # Find results
    mats = find_results(results, _rec_cg)
    n_mat = len(mats)

    # Find how many coefficients there are for output
    n_coef = None
    n_band = None
    for i, m in enumerate(mats):
        try:
            mat = spio.loadmat(m, squeeze_me=True,
                               struct_as_record=True)['rec_cg']
        except:
            continue

        try:
            n_coef, n_band = mat['coefs'][0].shape
        except:
            continue
        else:
            break
    if n_coef is None or n_band is None:
        raise Exception('Could not determine the number of bands')

    # Alias make_X to corresponding make_X# where # is n_coef
    if n_coef == 4:
        make_X = make_X4
    elif n_coef == 6:
        make_X = make_X6
    elif n_coef == 8:
        make_X = make_X8
    else:
        raise NotImplementedError('Supports 4/6/8 coefficients only')

    # Create X matrix from date
    X = make_X(date)

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
                     dtype=np.int16) * -9999

    for _i, m in enumerate(mats):
        # Verbose progress
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.0f}%'.format(_i / n_mat * 100))
        # Open MATLAB output
        try:
            mat = spio.loadmat(m, squeeze_me=True,
                               struct_as_record=True)['rec_cg']
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(f=m))
            continue

        if mat.ndim == 0:
            # No values in this .mat file
            continue

        # Find indices for the date specified
        if after is True:
            index = np.where(mat['t_end'] >= date)[0]
            _, _index = np.unique(mat['pos'][index], return_index=True)
            index = index[_index]
        else:
            index = np.where((mat['t_start'] <= date) &
                             (mat['t_end'] >= date))[0]

        if index.shape[0] == 0:
            continue

        # Locate entries in the map
        [rows, cols] = np.unravel_index(
            mat['pos'][index].astype(int) - 1,
            (image_ds.RasterYSize, image_ds.RasterXSize),
            order='C'
        )

        for i, r, c in itertools.izip(index, rows, cols):
            for i_b, b in enumerate(i_bands):
                # Calculate predicted image
                raster[r, c, i_b] = np.dot(mat['coefs'][i][:, b], X)

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
    ### Parse input
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

    # Output bands
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

    # Coefficient options
    coefs = [c for c in args['--coef'].replace(',', ' ').split(' ') if c != '']
    if not all([c.lower() in _coefs for c in coefs]):
        logger.error('Unknown coefficient options')
        sys.exit(1)

    # Go to next time segment option
    after = args['--after']
    before = args['--before']

    # NDV
    try:
        ndv = float(args['--ndv'])
    except ValueError:
        logger.error('NoDataValue must be a real number')
        raise

    # Root directory
    root = args['--directory']
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
            get_coefficients(date, after, bands, coefs, results, image_ds)
    elif args['predict']:
        raster = get_prediction(date, after, bands, results, image_ds)

    if args['class']:
        write_output(raster, output, image_ds, gdal_frmt, ndv, band_names)
    else:
        write_output(raster, output, image_ds, gdal_frmt, -9999, band_names)

    image_ds = None


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)

    main()
