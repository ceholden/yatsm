""" Functions related to writing to and retrieving from cache files
"""
import os

import numpy as np

from log_yatsm import logger

_image_ID_str = 'image_IDs'


def get_line_cache_name(dataset_config, n_images, row, nbands):
    """ Returns cache filename for specified config and line number

    Args:
        dataset_config (dict): configuration information about the dataset
        n_images (int): number of images in dataset
        row (int): line of the dataset for output
        nbands (int): number of bands in dataset

    Returns:
        str: filename of cache file

    """
    path = dataset_config.get('cache_line_dir')
    if not path:
        return

    filename = 'yatsm_r%i_n%i_b%i.npy.npz' % (row, n_images, nbands)

    return os.path.join(path, filename)


def get_line_cache_pattern(row, nbands, regex=False):
    """ Returns a pattern for a cache file from a certain row

    This function is useful for finding all cache files from a line, ignoring
    the number of images in the file.

    Args:
        row (int): line of the dataset for output
        nbands (int): number of bands in dataset
        regex (bool, optional): return a regular expression instead of glob
            style (default: False)

    Returns:
        str: filename pattern for cache files from line ``row``

    """
    wildcard = '.*' if regex else '*'
    pattern = 'yatsm_r{l}_n{w}_b{b}.npy.npz'.format(
        l=row, w=wildcard, b=nbands)

    return pattern


def test_cache(dataset_config):
    """ Test cache directory for ability to read from or write to

    Args:
        dataset_config (dict): dictionary of dataset configuration options

    Returns:
        tuple: tuple of bools describing ability to read from and write to
            cache directory

    """
    # Try to find / use cache
    read_cache = False
    write_cache = False

    cache_dir = dataset_config.get('cache_line_dir')
    if cache_dir:
        # Test existence
        if os.path.isdir(cache_dir):
            if os.access(cache_dir, os.R_OK):
                read_cache = True
            if os.access(cache_dir, os.W_OK):
                write_cache = True
            if read_cache and not write_cache:
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

    If ``image_IDs`` is not None this function will try to ensure data from
    cache file come from the list of image IDs provided. If cache file does not
    contain a list of image IDs, it will skip the check and return cache data.

    Args:
        cache_filename (str): cache filename
        image_IDs (iterable, optional): list of image IDs corresponding to data
            in cache file. If not specified, function will not check for
            correspondence (default: None)

    Returns:
        np.ndarray, or None: Return Y as np.ndarray if possible and if the
            cache file passes the consistency check specified by ``image_IDs``,
            else None

    """
    try:
        cache = np.load(cache_filename)
    except IOError:
        return None

    if _image_ID_str in cache.files and image_IDs is not None:
        if not np.array_equal(image_IDs, cache[_image_ID_str]):
            logger.warning('Cache file data in {f} do not match images '
                           'specified'.format(f=cache_filename))
            return None

    return cache['Y']


def write_cache_file(cache_filename, Y, image_IDs):
    """ Writes data to a cache file using np.savez_compressed

    Args:
        cache_filename (str): cache filename
        Y (np.ndarray): data to write to cache file
        image_IDs (iterable): list of image IDs corresponding to data in cache
            file. If not specified, function will not check for correspondence

    """
    np.savez_compressed(cache_filename, **{
        'Y': Y, _image_ID_str: image_IDs
    })


# Cache file updating
def update_cache_file(images, image_IDs,
                      old_cache_filename, new_cache_filename,
                      line, reader):
    """ Modify an existing cache file to contain data within `images`

    This should be useful for updating a set of cache files to reflect
    modifications to the timeseries dataset without completely reading the
    data into another cache file.

    For example, the cache file could be updated to reflect the deletion of
    a misregistered or cloudy image. Another common example would be for
    updating cache files to include newly acquired observations.

    Note that this updater will not handle updating cache files to include
    new bands.

    Args:
        images (iterable): list of new image filenames
        image_IDs (iterable): list of new image identifying strings
        old_cache_filename (str): filename of cache file to update
        new_cache_filename (str): filename of new cache file which includes
            modified data
        line (int): the line of data to be updated
        reader (callable): GDAL or BIP image reader function from
            :mod:`yatsm.io.stack_line_readers`

    Raises:
        ValueError: Raise error if old cache file does not record ``image_IDs``

    """
    images = np.asarray(images)
    image_IDs = np.asarray(image_IDs)

    # Cannot proceed if old cache file doesn't store filenames
    old_cache = np.load(old_cache_filename)
    if _image_ID_str not in old_cache.files:
        raise ValueError('Cannot update cache.'
                         'Old cache file does not store image IDs.')
    old_IDs = old_cache[_image_ID_str]
    old_Y = old_cache['Y']
    nband, _, ncol = old_Y.shape

    # Create new Y and add in values retained from old cache
    new_Y = np.zeros((nband, image_IDs.size, ncol),
                     dtype=old_Y.dtype.type)
    new_IDs = np.zeros(image_IDs.size, dtype=image_IDs.dtype)

    # Check deletions -- find which indices to retain in new cache
    retain_old = np.where(np.in1d(old_IDs, image_IDs))[0]
    if retain_old.size == 0:
        logger.warning('No image IDs in common in old cache file.')
    else:
        logger.debug('    retaining {r} of {n} images'.format(
            r=retain_old.size, n=old_IDs.size))
        # Find indices of old data to insert into new data
        idx_old_IDs = np.argsort(old_IDs)
        sorted_old_IDs = old_IDs[idx_old_IDs]
        idx_IDs = np.searchsorted(sorted_old_IDs,
                                  image_IDs[np.in1d(image_IDs, old_IDs)])

        retain_old = idx_old_IDs[idx_IDs]

        # Indices to insert into new data
        retain_new = np.where(np.in1d(image_IDs, old_IDs))[0]

        new_Y[:, retain_new, :] = old_Y[:, retain_old, :]
        new_IDs[retain_new] = old_IDs[retain_old]

    # Check additions -- find which indices we need to insert
    insert = np.where(np.in1d(image_IDs, old_IDs, invert=True))[0]

    if retain_old.size == 0 and insert.size == 0:
        raise ValueError('Cannot update cache file -- '
                         'no data retained or added')

    # Read in the remaining data from disk
    if insert.size > 0:
        logger.debug('Inserting {n} new images into cache'.format(
            n=insert.size))
        insert_Y = reader.read_row(images[insert], line)
        new_Y[:, insert, :] = insert_Y
        new_IDs[insert] = image_IDs[insert]

    np.testing.assert_equal(new_IDs, image_IDs)

    # Save
    write_cache_file(new_cache_filename, new_Y, image_IDs)
