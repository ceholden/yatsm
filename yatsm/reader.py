""" Functions for reading timeseries data
"""
from datetime import datetime as dt
import fnmatch
import logging
import os
import sys

import numpy as np
from osgeo import gdal, gdal_array

gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')


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


def read_pixel_timeseries(files, px, py):
    """ Returns NumPy array containing timeseries values for one pixel

    Args:
      files (list): List of filenames to read from
      px (int): Pixel X location
      py (int): Pixel Y location

    Returns:
      np.ndarray: Array (nband x n_images) containing all timeseries data
        from one pixel

    """
    nrow, ncol, nband, dtype = get_image_attribute(files[0])

    if px < 0 or px >= ncol or py < 0 or py >= nrow:
        raise IndexError('Row/column {r}/{c} is outside of image '
                         '(nrow/ncol: {nrow}/{ncol})'.format(
                            r=py, c=px, nrow=nrow, ncol=ncol))

    Y = np.zeros((nband, len(files)), dtype=dtype)

    for i, f in enumerate(files):
        ds = gdal.Open(f, gdal.GA_ReadOnly)
        for b in xrange(nband):
            Y[b, i] = ds.GetRasterBand(b + 1).ReadAsArray(px, py, 1, 1)

    return Y


def find_stack_images(location, folder_pattern='L*', image_pattern='L*stack',
                      date_index_start=9, date_index_end=16,
                      date_format='%Y%j',
                      ignore=['YATSM']):
    """ Find and identify dates and filenames of Landsat image stacks

    Args:
      location (str): Stacked image dataset location
      folder_pattern (str, optional): Filename pattern for stack image
        folders located within `location` (default: 'L*')
      image_pattern (str, optional): Filename pattern for stacked images
        located within each folder (default: 'L*stack')
      date_index_start (int, optional): Starting index of image date string
        within folder name (default: 9)
      date_index_end (int, optional): Ending index of image date string within
        folder name (default: 16)
      date_format (str, optional): String format of date within folder names
        (default: '%Y%j')
      ignore (list, optional): List of folder names within `location` to
        ignore from search (default: ['YATSM'])

    Returns:
      tuple: Tuple of lists containing the dates and filenames of all stacked
        images located

    """
    if isinstance(ignore, str):
        ignore = [ignore]

    folder_names = []
    image_filenames = []
    dates = []

    # Populate - only checking one directory down
    location = location.rstrip(os.path.sep)
    num_sep = location.count(os.path.sep)

    for root, dnames, fnames in os.walk(location, followlinks=True):
        # Remove results folder
        dnames[:] = [d for d in dnames for i in ignore if i not in d]

        # Force only 1 level
        num_sep_this = root.count(os.path.sep)
        if num_sep + 1 <= num_sep_this:
            del dnames[:]

        # Directory names as image IDs
        for dname in fnmatch.filter(dnames, folder_pattern):
            folder_names.append(dname)

        # Add file name and paths
        for fname in fnmatch.filter(fnames, image_pattern):
            image_filenames.append(os.path.join(root, fname))

    # Check to see if we found anything
    if not folder_names or not image_filenames:
        raise Exception('Zero stack images found with image '
                        'and folder patterns: {0}, {1}'.format(
                            folder_pattern, image_pattern))

    if len(folder_names) != len(image_filenames):
        raise Exception(
            'Inconsistent number of stacks folders and stack images located')

    # Extract dates
    for folder in folder_names:
        dates.append(
            dt.strptime(folder[date_index_start:date_index_end], date_format))

    # Sort images by date
    dates, image_filenames = (
        list(t) for t in
        zip(*sorted(zip(dates, image_filenames)))
    )

    return (dates, image_filenames)


class _BIPStackReader(object):
    """ Simple class to read BIP formatted stacks

    Some tests have shown that we can speed up total dataset read time by
    storing the file object references to each image as we loop over many rows
    instead of opening once per row read. This is a simple class designed to
    store these references.

    Note that this class assumes the images are "stacked" -- that is that all
    images contain the same number of rows, columns, and bands, and the images
    are of the same geographic extent.

    Args:
      filenames (list): list of filenames to read from
      size (tuple): tuple of (int, int) containing the number of columns and
        bands in the image
      datatype (np.dtype): NumPy datatype of the images

    Attributes:
      filenames (list): list of filenames to read from
      n_image (int): number of images
      size (tuple): tuple of (int, int) containing the number of columns and
        bands in the image
      datatype (np.dtype): NumPy datatype of images
      files (list): list of file objects for each image

    """
    def __init__(self, filenames, size, datatype):
        self.filenames = filenames
        self.n_image = len(self.filenames)
        self.size = size
        self.datatype = datatype

        self.files = []
        for f in self.filenames:
            self.files.append(open(f, 'rb'))

    def read_row(self, row):
        data = np.empty((self.size[1], self.n_image, self.size[0]),
                        self.datatype)

        for i, fid in enumerate(self.files):
            # Find where we need to seek to
            offset = np.dtype(self.datatype).itemsize * \
                (row * self.size[0]) * self.size[1]
            # Seek relative to current position
            fid.seek(offset - fid.tell(), 1)
            # Read
            data[:, i, :] = np.fromfile(fid,
                                        dtype=self.datatype,
                                        count=self.size[0] * self.size[1],
                                        ).reshape(self.size).T

        return data


_BIP_stack_reader = None
def read_row_BIP(filenames, row, size, dtype):
    """ Reads in an entire row of data from every image

    Args:
      filenames (iterable): sequence of filenames to read from
      row (int): row to read

    Returns:
      np.ndarray: 3D array (nband x nimage x ncol) containing the row
        of data

    """
    global _BIP_stack_reader
    if _BIP_stack_reader is None:
        _BIP_stack_reader = _BIPStackReader(filenames, size, dtype)

    return _BIP_stack_reader.read_row(row)


class _GDALStackReader(object):
    """ Simple class to read stacks using GDAL, keeping file objects open

    Some tests have shown that we can speed up total dataset read time by
    storing the file object references to each image as we loop over many rows
    instead of opening once per row read. This is a simple class designed to
    store these references.

    Note that this class assumes the images are "stacked" -- that is that all
    images contain the same number of rows, columns, and bands, and the images
    are of the same geographic extent.

    Args:
      filenames (list): list of filenames to read from

    Attributes:
      filenames (list): list of filenames to read from
      n_image (int): number of images
      n_band (int): number of bands in an image
      n_col (int): number of columns per row
      datatype (np.dtype): NumPy datatype of images
      datasets (list): list of GDAL datasets for all filenames
      dataset_bands (list): list of lists containing all GDAL raster band
        datasets, for all image filenames

    """
    def __init__(self, filenames):
        self.filenames = filenames

        self.datasets = []
        for f in self.filenames:
            self.datasets.append(gdal.Open(f, gdal.GA_ReadOnly))

        self.n_image = len(filenames)
        self.n_band = self.datasets[0].RasterCount
        self.n_col = self.datasets[0].RasterXSize
        self.datatype = gdal_array.GDALTypeCodeToNumericTypeCode(
            self.datasets[0].GetRasterBand(1).DataType)

        self.dataset_bands = []
        for ds in self.datasets:
            bands = []
            for i in xrange(self.n_band):
                bands.append(ds.GetRasterBand(i + 1))
            self.dataset_bands.append(bands)

    def read_row(self, row):
        """ Return a 3D NumPy array (nband x nimage x ncol) of one row's data

        Args:
          row (int): row in image to return

        Returns:
          np.ndarray: 3D NumPy array (nband x nimage x ncol) of image
            data for desired row

        """
        data = np.empty((self.n_band, self.n_image, self.n_col),
                        self.datatype)
        for i, ds_bands in enumerate(self.dataset_bands):
            for n_b, band in enumerate(ds_bands):
                data[n_b, i, :] = band.ReadAsArray(0, row, self.n_col, 1)

        return data


_gdal_stack_reader = None
def read_row_GDAL(filenames, row):
    """ Reads in an entire row of data from every image using GDAL

    Args:
      filenames (iterable): sequence of filenames to read from
      row (int): row to read

    Returns:
      np.ndarray: 3D array (nband x nimage x ncol) containing the row
        of data

    """
    global _gdal_stack_reader
    if _gdal_stack_reader is None:
        _gdal_stack_reader = _GDALStackReader(filenames)

    return _gdal_stack_reader.read_row(row)
