""" Functions related to writing to and retrieving from cache files
"""
import logging
import os

import numpy as np

# Log setup for runner
FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

_image_ID_str = 'image_IDs'


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
                logger.warning('Cache directory exists but is not writable')
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


def read_cache_file(cache_filename, image_IDs=None):
    """ Returns image data from a cache file

    If `image_IDs` is not None this function will try to ensure data from cache
    file come from the list of image IDs provided. If cache file does not
    contain a list of image IDs, it will skip the check and return cache data.

    Args:
      cache_filename (str): cache filename
      image_IDs (iterable, optional): list of image IDs corresponding to data
        in cache file. If not specified, function will not check for
        correspondence (default: None)

    Returns:
      np.ndarray, or None: Return Y as np.ndarray if possible and if the
        cache file passes the consistency check specified by `image_IDs`; else
        None

    """
    try:
        cache = np.load(cache_filename)
    except IOError:
        return None

    if _image_ID_str in cache.files and image_IDs is not None:
        if not all(image_IDs in cache[_image_ID_str]):
            return None

    return cache['Y']


def write_cache_file(cache_filename, Y, image_IDs):
    """ Writes data to a cache file using np.savez_compressed

    Args:
      cache_filename (str): cache filename
      Y (np.ndarray)
      image_IDs (iterable): list of image IDs corresponding to data in cache
        file. If not specified, function will not check for correspondence

    """
    np.savez_compressed(cache_filename, **{
        'Y': Y, _image_ID_str: image_IDs
    })
