#!/usr/bin/env python
""" Yet Another Timeseries Model (YATSM) - run script for lines of images

Usage: line_yatsm.py [options] <config_file> <job_number> <total_jobs>

Options:
    --check                     Check that images exist
    --resume                    Do not overwrite pre-existing results
    -v --verbose                Show verbose debugging messages
    --verbose-yatsm             Show verbose debugging messages in YATSM
    -q --quiet                  Show only error messages
    -h --help                   Show help

"""
from __future__ import division, print_function

import logging
import os
import sys
import time

from docopt import docopt

import numpy as np
from osgeo import gdal

# Handle runnin as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.config_parser import parse_config_file
from yatsm.utils import (calculate_lines, get_output_name, get_line_cache_name,
                         csvfile_to_dataset, make_X)
from yatsm.reader import get_image_attribute
from yatsm.yatsm import YATSM, TSLengthException

# Log setup for runner
FORMAT = '%(asctime)s %(levelname)s %(module)s.%(funcName)s : %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

# Logging level for YATSM
loglevel_YATSM = logging.WARNING


def test_cache(dataset_config):
    """ Test cache directory for ability to read from or write to

    Args:
      dataset_config (dict): dictionary of dataset configuration options

    Returns:
      (read_cache, write_cache): tuple of bools describing ability to read from
        and write to cache directory

    """
    # Try to find / use cache
    read_cache = False
    write_cache = False

    cache_dir = dataset_config.get('cache_line_dir')
    if cache_dir:
        # Test existence
        if os.path.isdir(cache_dir):
            read_cache = True
            if os.access(cache_dir, os.W_OK):
                write_cache = True
            else:
                logger.warning('Cache directory exist but is not writable')
        else:
            # If it doesn't already exist, can we create it?
            try:
                os.makedirs(cache_dir)
            except:
                logger.warning('Could not create cache directory')
            else:
                read_cache = True
                write_cache = True

    logger.debug('Attempt reading in from cache directory?: {b}'.format(
        b=read_cache))
    logger.debug('Attempt writing to cache directory?: {b}'.format(
        b=write_cache))

    return read_cache, write_cache


def read_line(line, images, dataset_config,
              ncol, nband, dtype,
              read_cache=False, write_cache=False):
    """ Reads in dataset from cache or images if required

    Args:
      line (int): line to read in from images
      images (list): list of image filenames to read from
      dataset_config (dict): dictionary of dataset configuration options
      ncol (int): number of columns
      nband (int): number of bands
      dtype (type): NumPy datatype
      read_cache (bool, optional): try to read from cache directory
      write_cache (bool, optional): to to write to cache directory

    Returns:
      Y (np.ndarray): 3D array of image data (nband, n_image, n_cols)

    """
    start_time = time.time()

    read_from_disk = True
    cache_filename = get_line_cache_name(
        dataset_config, len(images), line, nband)

    if read_cache:
        if os.path.isfile(cache_filename):
            logger.debug('Reading in Y from cache file {f}'.format(
                f=cache_filename))
            Y = np.load(cache_filename)['Y']
            read_from_disk = False

    if read_from_disk:
        # Read in Y
        Y = np.zeros((nband, len(images), ncol), dtype=dtype)

        # TODO: implement BIP reader
        # Read in data just using GDAL
        logger.debug('Reading in data from disk')
        for i, image in enumerate(images):
            ds = gdal.Open(image, gdal.GA_ReadOnly)
            for b in xrange(ds.RasterCount):
                Y[b, i, :] = ds.GetRasterBand(b + 1).ReadAsArray(
                    0, line, ncol, 1)

    if write_cache and read_from_disk:
        logger.debug('Writing Y data to cache file {f}'.format(
            f=cache_filename))
        np.savez_compressed(cache_filename, Y=Y)

    logger.debug('Took {s}s to read in the data'.format(
        s=round(time.time() - start_time, 2)))

    return Y


# Runner
def run_line(line, X, images,
             dataset_config, yatsm_config,
             nrow, ncol, nband, dtype,
             read_cache=False, write_cache=False):
    """ Runs YATSM for a line

    Args:
      line (int): line to be run from image
      dates (ndarray): np.array of X feature from ordinal dates
      images (ndarray): np.array of image filenames
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
      nrow (int): number of rows
      ncol (int): number of columns
      nband (int): number of bands
      dtype (type): NumPy datatype
      read_cache (bool, optional): try to read from cache directory
      write_cache (bool, optional): to to write to cache directory

    """
    # Setup output
    output = []

    Y = read_line(line, images, dataset_config,
                  ncol, nband, dtype,
                  read_cache=read_cache, write_cache=write_cache)

    # About to run YATSM
    logger.debug('    running YATSM')
    # Raise or lower logging level for YATSM
    _level = logger.level
    logger.setLevel(loglevel_YATSM)

    for c in xrange(Y.shape[-1]):
        try:
            result = run_pixel(X, Y[..., c], dataset_config, yatsm_config,
                               px=c, py=line)
        except TSLengthException:
            continue

        output.extend(result)

    # Return logging level
    logger.setLevel(_level)

    # Save output
    outfile = get_output_name(dataset_config, line)
    logger.debug('    saving YATSM output to {f}'.format(f=outfile))

    np.savez(outfile,
             version=__version__,
             consecutive=yatsm_config['consecutive'],
             threshold=yatsm_config['threshold'],
             min_obs=yatsm_config['min_obs'],
             min_rmse=yatsm_config['min_rmse'],
             retrain_time=yatsm_config['retrain_time'],
             screening=yatsm_config['screening'],
             screening_crit=yatsm_config['screening_crit'],
             lassocv=yatsm_config['lassocv'],
             reverse=yatsm_config['reverse'],
             robust=yatsm_config['robust'],
             freq=yatsm_config['freq'],
             record=np.array(output))


def run_pixel(X, Y, dataset_config, yatsm_config, px=0, py=0):
    """ Run a single pixel through YATSM

    Args:
      X (ndarray): 2D (nimage x nband) feature input from ordinal date
      Y (ndarray): 2D (nband x nimage) image input
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
      px (int, optional):       X (column) pixel reference
      py (int, optional):       Y (row) pixel reference

    Returns:
      model_result (ndarray): NumPy array of model results from YATSM

    """
    # Mask
    mask_band = dataset_config['mask_band']

    # Continue if clear observations are less than 50% of dataset
    if (Y[mask_band, :] < 255).sum() < Y.shape[1] / 2.0:
        raise TSLengthException('Not enough valid observations')

    # Otherwise continue
    clear = np.logical_and(Y[mask_band, :] <= 1,
                           np.all(Y <= 10000, axis=0))
    Y = Y[:mask_band, clear]
    X = X[clear, :]

    if yatsm_config['reverse']:
        # TODO: do this earlier
        X = np.flipud(X)
        Y = np.fliplr(Y)

    yatsm = YATSM(X, Y,
                  consecutive=yatsm_config['consecutive'],
                  threshold=yatsm_config['threshold'],
                  min_obs=yatsm_config['min_obs'],
                  min_rmse=yatsm_config['min_rmse'],
                  test_indices=yatsm_config['test_indices'],
                  retrain_time=yatsm_config['retrain_time'],
                  screening=yatsm_config['screening'],
                  screening_crit=yatsm_config['screening_crit'],
                  green_band=dataset_config['green_band'] - 1,
                  swir1_band=dataset_config['swir1_band'] - 1,
                  lassocv=yatsm_config['lassocv'],
                  px=px,
                  py=py,
                  logger=logger)
    yatsm.run()

    if yatsm_config['robust']:
        return yatsm.robust_record
    else:
        return yatsm.record


def main(dataset_config, yatsm_config,
         check=False, resume=False,
         read_cache=False, write_cache=False):
    """ Read in dataset and YATSM for a complete line

    Args:
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
      check (bool, optional): check to make sure images are readible
      resume (bool, optional): do not overwrite existing results, instead
        continue from first non-existing result file
      read_cache (bool, optional): try to read from cache directory
      write_cache (bool, optional): to to write to cache directory

    """
    # Read in dataset
    dates, images = csvfile_to_dataset(
        dataset_config['input_file'],
        date_format=dataset_config['date_format']
    )

    # Check for existence of files and remove missing
    if check:
        to_delete = []
        for i, img in enumerate(images):
            if not os.path.isfile(img):
                logger.warning('Could not find file {f} -- removing'.
                               format(f=img))
                to_delete.append(i)

        if len(to_delete) == 0:
            logger.debug('Checked and found all input images')
        else:
            logger.warning('Removing {n} images'.format(n=len(to_delete)))
            dates = np.delete(dates, np.array(to_delete))
            images = np.delete(images, np.array(to_delete))

    # Get attributes of one of the images
    nrow, ncol, nband, dtype = get_image_attribute(images[0])

    # Calculate the lines this job ID works on
    job_lines = calculate_lines(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Calculate X feature input
    X = make_X(dates, yatsm_config['freq']).T

    # Start running YATSM
    start_time_all = time.time()
    logger.info('Starting to run lines')
    for job_line in job_lines:
        if resume:
            try:
                z = np.load(get_output_name(dataset_config, job_line))
            except:
                pass
            else:
                del z
                logger.debug('Already processed line {l}'.format(l=job_line))
                continue

        logger.debug('Running line {l}'.format(l=job_line))
        start_time = time.time()

        try:
            run_line(job_line, X, images,
                     dataset_config, yatsm_config,
                     nrow, ncol, nband, dtype,
                     read_cache=read_cache, write_cache=write_cache)
        except Exception as e:
            logger.error('Could not process line {l}'.format(l=job_line))
            logger.error(str(e))

        logger.debug('Took {s}s to run'.format(
            s=round(time.time() - start_time, 2)))

    logger.info('Completed {n} lines in {m} minutes'.format(
        n=len(job_lines),
        m=round((time.time() - start_time_all) / 60.0, 2)
    ))


if __name__ == '__main__':
    # Get arguments
    args = docopt(__doc__, version=__version__)

    # Validate input arguments
    config_file = args['<config_file>']
    if not os.path.isfile(args['<config_file>']):
        print('Error - <config_file> specified is not a file')
        sys.exit(1)

    try:
        job_number = int(args['<job_number>'])
    except:
        print('Error - <job_number> must be an integer greater than 0')
        sys.exit(1)
    if job_number <= 0:
        print('Error - <job_number> cannot be less than or equal to 0')
        sys.exit(1)
    job_number -= 1

    try:
        total_jobs = int(args['<total_jobs>'])
    except:
        print('Error - <total_jobs> must be an integer')
        sys.exit(1)

    # Check for existence of images?
    check = args['--check']

    # Resume?
    resume = False
    if args['--resume']:
        resume = True

    # Setup logger
    if args['--verbose']:
        logger.setLevel(logging.DEBUG)

    if args['--verbose-yatsm']:
        loglevel_YATSM = logging.DEBUG

    if args['--quiet']:
        loglevel_YATSM = logging.WARNING
        logger.setLevel(logging.WARNING)

    # Parse and validate configuration file
    dataset_config, yatsm_config = parse_config_file(config_file)

    # Make output directory
    try:
        os.makedirs(dataset_config['output'])
    except OSError as e:
        # File exists
        if e.errno == 17:
            pass
        elif e.errno == 13:
            print('Error - cannot create output directory {d}'.format(
                d=dataset_config['output']))
            print(e.strerror)
            sys.exit(1)

    # Test write capability
    if not os.access(dataset_config['output'], os.W_OK):
        print('Error - cannot write to output directory {d}'.format(
            d=dataset_config['output']))
        sys.exit(1)

    # Test existence of cache directory
    read_cache, write_cache = test_cache(dataset_config)

    # Run YATSM
    logger.info('Job {i} / {n} - using config file {f}'.format(
                i=job_number, n=total_jobs, f=config_file))
    main(dataset_config, yatsm_config, check=check, resume=resume,
         read_cache=read_cache, write_cache=write_cache)
