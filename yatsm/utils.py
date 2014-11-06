import csv
from datetime import datetime as dt
import logging
import os

import numpy as np
from osgeo import gdal
from osgeo import gdal_array

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

