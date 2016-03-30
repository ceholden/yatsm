""" Helper functions for reading various types of imagery data
"""
import logging
import time

import numpy as np
from osgeo import gdal, gdal_array

from .stack_line_readers import bip_reader, gdal_reader
from .. import cache

logger = logging.getLogger('yatsm')


def get_image_attribute(image_filename):
    """ Use GDAL to open image and return some attributes

    Args:
        image_filename (str): image filename

    Returns:
        tuple: nrow (int), ncol (int), nband (int), NumPy datatype (type)
    """
    try:
        image_ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    except Exception as e:
        logger.error('Could not open example image dataset ({f}): {e}'
                     .format(f=image_filename, e=str(e)))
        raise

    nrow = image_ds.RasterYSize
    ncol = image_ds.RasterXSize
    nband = image_ds.RasterCount
    dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
        image_ds.GetRasterBand(1).DataType)

    return (nrow, ncol, nband, dtype)


def read_image(image_filename, bands=None, dtype=None):
    """ Return raster image bands as a sequence of NumPy arrays

    Args:
        image_filename (str): Image filename
        bands (iterable, optional): A sequence of bands to read from image.
            If `bands` is None, function returns all bands in raster. Note that
            bands are indexed on 1 (default: None)
        dtype (np.dtype): NumPy datatype to use for image bands. If `dtype` is
            None, arrays are kept as the image datatype (default: None)

    Returns:
        list: list of NumPy arrays for each band specified

    Raises:
        IOError: raise IOError if bands specified are not contained within
            raster
        RuntimeError: raised if GDAL encounters errors
    """
    try:
        ds = gdal.Open(image_filename, gdal.GA_ReadOnly)
    except:
        logger.error('Could not read image {i}'.format(i=image_filename))
        raise

    if bands:
        if not all([b in range(1, ds.RasterCount + 1) for b in bands]):
            raise IOError('Image {i} ({n} bands) does not contain bands '
                          'specified (requested {b})'.
                          format(i=image_filename, n=ds.RasterCount, b=bands))
    else:
        bands = range(1, ds.RasterCount + 1)

    if not dtype:
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
            ds.GetRasterBand(1).DataType)

    output = []
    for b in bands:
        output.append(ds.GetRasterBand(b).ReadAsArray().astype(dtype))

    return output


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
                         '(nrow/ncol: {nrow}/{ncol})'.
                         format(r=py, c=px, nrow=nrow, ncol=ncol))

    Y = np.zeros((nband, len(files)), dtype=dtype)

    for i, f in enumerate(files):
        ds = gdal.Open(f, gdal.GA_ReadOnly)
        for b in range(nband):
            Y[b, i] = ds.GetRasterBand(b + 1).ReadAsArray(px, py, 1, 1)

    return Y


def read_line(line, images, image_IDs, dataset_config,
              ncol, nband, dtype,
              read_cache=False, write_cache=False, validate_cache=False):
    """ Reads in dataset from cache or images if required

    Args:
        line (int): line to read in from images
        images (list): list of image filenames to read from
        image_IDs (iterable): list image identifying strings
        dataset_config (dict): dictionary of dataset configuration options
        ncol (int): number of columns
        nband (int): number of bands
        dtype (type): NumPy datatype
        read_cache (bool, optional): try to read from cache directory
            (default: False)
        write_cache (bool, optional): try to to write to cache directory
            (default: False)
        validate_cache (bool, optional): validate that cache data come from
            same images specified in `images` (default: False)

    Returns:
        np.ndarray: 3D array of image data (nband, n_image, n_cols)
    """
    start_time = time.time()

    read_from_disk = True
    cache_filename = cache.get_line_cache_name(
        dataset_config, len(images), line, nband)

    Y_shape = (nband, len(images), ncol)

    if read_cache:
        Y = cache.read_cache_file(cache_filename,
                                  image_IDs if validate_cache else None)
        if Y is not None and Y.shape == Y_shape:
            logger.debug('Read in Y from cache file')
            read_from_disk = False
        elif Y is not None and Y.shape != Y_shape:
            logger.warning(
                'Data from cache file does not meet size requested '
                '({y} versus {r})'.format(y=Y.shape, r=Y_shape))

    if read_from_disk:
        # Read in Y
        if dataset_config['use_bip_reader']:
            # Use BIP reader
            logger.debug('Reading in data from disk using BIP reader')
            Y = bip_reader.read_row(images, line)
        else:
            # Read in data just using GDAL
            logger.debug('Reading in data from disk using GDAL')
            Y = gdal_reader.read_row(images, line)

        logger.debug('Took {s}s to read in the data'.format(
            s=round(time.time() - start_time, 2)))

    if write_cache and read_from_disk:
        logger.debug('Writing Y data to cache file {f}'.format(
            f=cache_filename))
        cache.write_cache_file(cache_filename, Y, image_IDs)

    return Y
