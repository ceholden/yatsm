from __future__ import division

from datetime import datetime as dt
import fnmatch
import os
import sys

import numpy as np
import pandas as pd

try:
    from scandir import walk
except:
    from os import walk

from log_yatsm import logger


# JOB SPECIFIC FUNCTIONS
def distribute_jobs(job_number, total_jobs, n, interlaced=True):
    """ Assign `job_number` out of `total_jobs` a subset of `n` tasks

    Args:
      job_number (int): 0-indexed processor to distribute jobs to
      total_jobs (int): total number of processors running jobs
      n (int): number of tasks (e.g., lines in image, regions in segment)
      interlaced (bool, optional): interlace job assignment (default: True)

    Returns:
      np.ndarray: np.ndarray of task IDs to be processed

    Raises:
      ValueError: raise error if `job_number` and `total_jobs` specified
        result in no jobs being assinged (happens if `job_number` and
        `total_jobs` are both 1)

    """
    if interlaced:
        assigned = 0
        tasks = []

        while job_number + total_jobs * assigned < n:
            tasks.append(job_number + total_jobs * assigned)
            assigned += 1
        tasks = np.asarray(tasks)
    else:
        size = int(n / total_jobs)
        i_start = size * job_number
        i_end = size * (job_number + 1)

        tasks = np.arange(i_start, min(i_end, n))

    if tasks.size == 0:
        raise ValueError(
            'No jobs assigned for job_number/total_jobs: {j}/{t}'.format(
                j=job_number,
                t=total_jobs))

    return tasks


def get_output_name(dataset_config, line):
    """ Returns output name for specified config and line number

    Args:
      dataset_config (dict): configuration information about the dataset
      line (int): line of the dataset for output

    Returns:
      filename (str): output filename

    """
    return os.path.join(dataset_config['output'],
                        '%s%s.npz' % (dataset_config['output_prefix'], line))


# IMAGE DATASET READING
def csvfile_to_dataframe(input_file, date_format='%Y%j'):
    """ Return sorted filenames of images from input text file

    Args:
      input_file (str): text file of dates and files
      date_format (str): format of dates in file

    Returns:
      pd.DataFrame: pd.DataFrame of dates, sensor IDs, and filenames

    """
    df = pd.read_csv(input_file)

    # Guess and convert date field
    date_col = [i for i, n in enumerate(df.columns) if 'date' in n.lower()]
    if not date_col:
        raise KeyError('Could not find date column in input file')
    if len(date_col) > 1:
        logger.warning('Multiple date columns found in input CSV file. '
                       'Using %s' % df.columns[date_col[0]])
    date_col = df.columns[date_col[0]]

    df[date_col] = pd.to_datetime(
        df[date_col], format=date_format).map(lambda x: dt.toordinal(x))

    return df


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

    driver = gdal.GetDriverByName(str(gdal_frmt))

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
            ds.GetRasterBand(1).SetMetadata({'band_1': band_names[0]})

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
    for root, dirnames, filenames in walk(location):
        for filename in fnmatch.filter(filenames, pattern):
            records.append(os.path.join(root, filename))

    if len(records) == 0:
        raise IOError('Could not find results in: %s' % location)

    records.sort()

    return records


def iter_records(records, warn_on_empty=False, yield_filename=False):
    """ Iterates over records, returning result NumPy array

    Args:
      records (list): List containing filenames of results
      warn_on_empty (bool, optional): Log warning if result contained no
        result records (default: False)
      yield_filename (bool, optional): Yield the filename and the record

    Yields:
      np.ndarray or tuple: Result saved in record and the filename, if desired

    """
    n_records = len(records)

    for _i, r in enumerate(records):
        # Verbose progress
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.1f}%'.format(_i / n_records * 100))
        # Open output
        try:
            rec = np.load(r)['record']
        except (ValueError, AssertionError, IOError) as e:
            logger.warning('Error reading a result file (may be corrupted) (%s): %s' % (r, e.message))
            continue

        if rec.shape[0] == 0:
            # No values in this file
            if warn_on_empty:
                logger.warning('Could not find results in {f}'.format(f=r))
            continue

        if yield_filename:
            yield rec, r
        else:
            yield rec


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
