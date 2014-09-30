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

# Support namechange in Python3
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import csv
from datetime import datetime as dt
import logging
import os
import sys
import time

from docopt import docopt

import numpy as np
from osgeo import gdal
from osgeo import gdal_array

from version import __version__
from yatsm import make_X, YATSM, TSLengthException


# Log setup for runner
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Logging level for YATSM
loglevel_YATSM = logging.WARNING


# JOB SPECIFIC FUNCTIONS
def calculate_lines(nrow):
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
    """ Returns output name for specified config and line number """
    return os.path.join(dataset_config['output'],
                        'yatsm_r{line}'.format(line=line) + '.npz')


# IMAGE DATASET READING
def find_images(input_file, date_format='%Y-%j'):
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
                logger.error('Input config file: {f}'.format(f=config))
                logger.error('Input dataset file: {f}'.format(f=input_file))
                logger.error('Date format: {f}'.format(f=date_format))
                raise
                sys.exit(1)
            else:
                i_date = 1
                i_image = 0

        f.seek(0)

        logger.debug('Reading in image date and filenames')
        for row in reader:
            dates.append(dt.strptime(row[i_date], date_format).toordinal())
            images.append(row[i_image])

        return (np.array(dates), np.array(images))


def get_image_attribute(image_filename):
    """ Use GDAL to open image and return some attributes

    Args:
      image_filename (string): image filename

    Returns:
      tuple (int, int, int, type): nrow, ncol, nband, NumPy datatype

    """
    try:
        image_ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image dataset ({f})'.format(
            f=image_filename))
        sys.exit(1)

    nrow = image_ds.RasterYSize
    ncol = image_ds.RasterXSize
    nband = image_ds.RasterCount
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
        image_ds.GetRasterBand(1).DataType)

    return (nrow, ncol, nband, dtype)


# CONFIG FILE PARSING
def _parse_config_v_zero_pt_one(config):
    """ Parses config file for version 0.1.x """
    # Configuration for dataset
    dataset_config = dict.fromkeys(['input_file', 'date_format',
                                    'output',
                                    'n_bands', 'mask_band',
                                    'green_band', 'swir1_band',
                                    'use_bip_reader'])
    for k in dataset_config:
        dataset_config[k] = config.get('dataset', k)

    dataset_config['n_bands'] = int(dataset_config['n_bands'])
    dataset_config['mask_band'] = int(dataset_config['mask_band']) - 1
    dataset_config['green_band'] = int(dataset_config['green_band']) - 1
    dataset_config['swir1_band'] = int(dataset_config['swir1_band']) - 1

    # Configuration for YATSM algorithm
    yatsm_config = {}

    yatsm_config['consecutive'] = int(config.get('YATSM', 'consecutive'))
    yatsm_config['threshold'] = float(config.get('YATSM', 'threshold'))
    yatsm_config['min_obs'] = int(config.get('YATSM', 'min_obs'))
    yatsm_config['min_rmse'] = float(config.get('YATSM', 'min_rmse'))
    yatsm_config['freq'] = config.get(
        'YATSM', 'freq').replace(',', ' ').split(' ')
    yatsm_config['freq'] = [int(v) for v in yatsm_config['freq']
                            if v != '']
    yatsm_config['test_indices'] = config.get(
        'YATSM', 'test_indices').replace(',', ' ').split(' ')
    yatsm_config['test_indices'] = np.array([int(b) for b in
                                             yatsm_config['test_indices']
                                             if b != ''])
    yatsm_config['screening'] = config.get('YATSM', 'screening')
    yatsm_config['lassocv'] = config.get('YATSM', 'lassocv').lower() == 'true'
    yatsm_config['reverse'] = config.get('YATSM', 'reverse').lower() == 'true'
    yatsm_config['robust'] = config.get('YATSM', 'robust').lower() == 'true'

    return (dataset_config, yatsm_config)


def parse_config_file(config):
    """ Parses config file into dictionary of attributes """

    dataset_config = None
    yatsm_config = None

    # Parse different versions
    version = config.get('metadata', 'version').split('.')

    # 0.1.x
    if version[0] == '0' and version[1] == '1':
        dataset_config, yatsm_config = _parse_config_v_zero_pt_one(config)

    return (dataset_config, yatsm_config)


# Runner
def run_line(line, X, images,
             dataset_config, yatsm_config,
             nrow, ncol, nband, dtype,
             use_BIP=False):
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
      use_BIP (bool, optional): use BIP line reader

    """
    # Setup output
    output = []

    # Read in Y
    Y = np.zeros((nband, len(images), ncol), dtype=dtype)

    # TODO: implement BIP reader
    if use_BIP:
        pass

    # Read in data just using GDAL
    logger.debug('    reading in data')
    for i, image in enumerate(images):
        ds = gdal.Open(image, gdal.GA_ReadOnly)
        for b in xrange(ds.RasterCount):
            Y[b, i, :] = ds.GetRasterBand(b + 1).ReadAsArray(0, line, ncol, 1)

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
             screening=yatsm_config['screening'],
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
                  lassocv=yatsm_config['lassocv'],
                  screening=yatsm_config['screening'],
                  px=px,
                  py=py,
                  logger=logger)
    yatsm.run()

    if yatsm_config['robust']:
        return yatsm.robust_record
    else:
        return yatsm.record


def main(dataset_config, yatsm_config, check=False, resume=False):
    """ Read in dataset and YATSM for a complete line """
    # Read in dataset
    dates, images = find_images(dataset_config['input_file'],
                                date_format=dataset_config['date_format'])

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
    job_lines = calculate_lines(nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Calculate X feature input
    X = make_X(dates, yatsm_config['freq']).T

    # Start running YATSM
    logger.info('Starting to run lines')
    for job_line in job_lines:
        if resume:
            try:
                z = np.load(get_output_name(dataset_config, job_line))
            except:
                pass
            else:
                del z
                logger.info('Already processed line {l}'.format(l=job_line))
                continue
        logger.debug('Running line {l}'.format(l=job_line))
        start_time = time.time()
        try:
            run_line(job_line, X, images,
                     dataset_config, yatsm_config,
                     nrow, ncol, nband, dtype,
                     use_BIP=dataset_config['use_bip_reader'])
        except:
            logger.error('Could not process line {l}'.format(l=job_line))
            raise
        logger.debug('Took {s}s to run'.format(
                s=round(time.time() - start_time, 2)))


if __name__ == '__main__':
    # Get arguments
    args = docopt(__doc__,
                  version=__version__)

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
    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_config, yatsm_config = parse_config_file(config)

    # Make output directory
    try:
        os.makedirs(dataset_config['output'])
    except:
        if os.path.isdir(dataset_config['output']):
            pass
        else:
            raise

    # Run YATSM
    main(dataset_config, yatsm_config, check=check, resume=resume)
