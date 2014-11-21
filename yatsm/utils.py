from __future__ import division

import csv
from datetime import datetime as dt
import fnmatch
import logging
import os
import sys

import numpy as np

logger = logging.getLogger('yatsm')


# JOB SPECIFIC FUNCTIONS
def calculate_lines(job_number, total_jobs, nrow):
    """ Calculate the lines this job processes given nrow, njobs, and job ID

    Args:
      nrow (int): number of rows in image

    Returns:
      rows (ndarray): np.array of rows to be processed

    """
    assigned = 0
    rows = []

    while job_number + total_jobs * assigned < nrow:
        rows.append(job_number + total_jobs * assigned)
        assigned += 1

    return np.array(rows)


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


def get_line_cache_name(dataset_config, n_images, nrow, nbands):
    """ Returns cache filename for specified config and line number

    Args:
      dataset_config (dict): configuration information about the dataset
      n_images (int): number of images in dataset
      nrow (int): line of the dataset for output
      nbands (int): number of bands in dataset

    Returns:
      str: filename of cache file

    """
    path = dataset_config['cache_line_dir']
    filename = 'yatsm_r{l}_n{n}_b{b}.npy.npz'.format(
        l=nrow, n=n_images, b=nbands)

    return os.path.join(path, filename)


# IMAGE DATASET READING
def csvfile_to_dataset(input_file, date_format='%Y-%j'):
    """ Return sorted filenames of images from input text file

    Args:
      input_file (str): text file of dates and files
      date_format (str): format of dates in file

    Returns:
      (ndarray, ndarray): paired dates and filenames of stacked images

    """
    # Store index of date and image
    i_date = 0
    i_image = 1

    dates = []
    images = []

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
                dt.strptime(row[i_image], date_format).toordinal()
            except:
                logger.debug('Could not parse second column to ordinal date')
                logger.error('Could not parse any columns to ordinal date')
                logger.error('Input dataset file: {f}'.format(f=input_file))
                logger.error('Date format: {f}'.format(f=date_format))
                raise
            else:
                i_date = 1
                i_image = 0

        f.seek(0)

        logger.debug('Reading in image date and filenames')
        for row in reader:
            dates.append(dt.strptime(row[i_date], date_format).toordinal())
            images.append(row[i_image])

        return (np.array(dates), np.array(images))


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
def is_integer(s):
    """ Returns True if `s` is an integer """
    try:
        int(s)
        return True
    except:
        return False
