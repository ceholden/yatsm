from __future__ import division

import csv
from datetime import datetime as dt
import fnmatch
import os
import sys

import numpy as np

from log_yatsm import logger


# JOB SPECIFIC FUNCTIONS
def calculate_lines(job_number, total_jobs, nrow, interlaced=True):
    """ Calculate the lines this job processes given nrow, njobs, and job ID

    Args:
      job_number (int): processor to distribute jobs to
      total_jobs (int): total number of processors running jobs
      nrow (int): number of rows in image
      interlaced (bool, optional): interlace line assignment (default: True)

    Returns:
      rows (np.ndarray): np.ndarray of rows to be processed

    """
    if interlaced:
        assigned = 0
        rows = []

        while job_number + total_jobs * assigned < nrow:
            rows.append(job_number + total_jobs * assigned)
            assigned += 1
        rows = np.array(rows)
    else:
        size = int(nrow / total_jobs) + 1
        i_start = size * job_number
        i_end = size * (job_number + 1)

        rows = np.arange(i_start, min(i_end, nrow))

    return rows


def get_output_name(dataset_config, line):
    """ Returns output name for specified config and line number

    Args:
      dataset_config (dict): configuration information about the dataset
      line (int): line of the dataset for output

    Returns:
      filename (str): output filename

    """
    return os.path.join(dataset_config['output'],
                        '{pref}{line}.npz'.format(
                            pref=dataset_config['output_prefix'],
                            line=line))


# IMAGE DATASET READING
def csvfile_to_dataset(input_file, date_format='%Y-%j'):
    """ Return sorted filenames of images from input text file

    Args:
      input_file (str): text file of dates and files
      date_format (str): format of dates in file

    Returns:
      (ndarray, ndarray, ndarray): dates, sensor IDs, and filenames of stacked
        images

    """
    # Store index of date and image
    i_date = 0
    i_sensor = 1
    i_image = 2

    dates = []
    images = []
    sensors = []

    logger.debug('Opening image dataset file')
    with open(input_file, 'rb') as f:
        reader = csv.reader(f)

        # Figure out which index is for what
        row = reader.next()

        try:
            dt.strptime(row[i_date], date_format).toordinal()
        except:
            logger.debug('Could not parse first column to ordinal date')
            try:
                dt.strptime(row[i_sensor], date_format).toordinal()
            except:
                logger.debug('Could not parse second column to ordinal date')
                logger.error('Could not parse any columns to ordinal date')
                logger.error('Input dataset file: {f}'.format(f=input_file))
                logger.error('Date format: {f}'.format(f=date_format))
                raise
            else:
                i_date = 1
                i_sensor = 0

        f.seek(0)

        logger.debug('Reading in image date, sensor, and filenames')
        for row in reader:
            dates.append(dt.strptime(row[i_date], date_format).toordinal())
            sensors.append(row[i_sensor])
            images.append(row[i_image])


        return (np.array(dates), np.array(sensors), np.array(images))


def get_image_IDs(filenames):
    """ Returns image IDs for each filename (basename of dirname of file)

    Args:
      filenames (iterable): filenames to return image IDs for

    Returns:
      list: image IDs for each file in `filenames`

    """
    return [os.path.basename(os.path.dirname(f)) for f in filenames]


# MAPPING UTILITIES
def write_output(raster, output, image_ds, gdal_frmt, ndv, band_names=None):
    """ Write raster to output file """
    from osgeo import gdal, gdal_array

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
                ds.GetRasterBand(b + 1).SetMetadata({
                    'band_{i}'.format(i=b + 1): band_names[b]
                })
    else:
        logger.debug('    writing band')
        ds.GetRasterBand(1).WriteArray(raster)
        ds.GetRasterBand(1).SetNoDataValue(ndv)

        if band_names is not None:
            ds.GetRasterBand(1).SetDescription(band_names[0])
            ds.GetRasterBand(1).SetMetadata({
                'band_1': band_names[0]
            })

    ds.SetProjection(image_ds.GetProjection())
    ds.SetGeoTransform(image_ds.GetGeoTransform())

    ds = None


# RESULT UTILITIES
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


def iter_records(records, warn_on_empty=False):
    """ Iterates over records, returning result NumPy array

    Args:
      records (list): List containing filenames of results
      warn_on_empty (bool, optional): Log warning if result contained no
        result records (default: False)

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
            if warn_on_empty:
                logger.warning('Could not find results in {f}'.format(f=r))
            continue

        yield rec


# CALCULATION UTILITIES
w = 2 * np.pi / 365.25


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


# MISC UTILITIES
def date2index(dates, d):
    """ Returns index of sorted array `dates` containing the date `d`

    Args:
      dates (np.ndarray): array of dates (or numbers really) in sorted order
      d (int, float): number to search for

    Returns:
      int: index of `dates` containing value `d`

    """
    return np.searchsorted(dates, d, side='right')


def is_integer(s):
    """ Returns True if `s` is an integer """
    try:
        int(s)
        return True
    except:
        return False
