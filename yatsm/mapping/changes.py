""" Functions relevant for mapping abrupt changes
"""
from datetime import datetime as dt
import logging

import numpy as np

from ..utils import find_results, iter_records

logger = logging.getLogger('yatsm')


def get_magnitude_indices(results):
    """ Finds indices of result containing magnitude of change information

    Args:
      results (iterable): list of result files to check within

    Returns:
      np.ndarray: indices containing magnitude change information from the
        tested indices

    Raises:
        KeyError: Raise KeyError when a required result output is missing
            from the saved record structure

    """
    for result in results:
        try:
            rec = np.load(result)
        except (ValueError, AssertionError) as e:
            logger.warning('Error reading %s. May be corrupted: %s' %
                           (result, e.message))
            continue

        # First search for record of `test_indices`
        if 'test_indices' in rec.files:
            logger.debug('Using `test_indices` information for magnitude')
            return rec['test_indices']

        # Fall back to using non-zero elements of 'record' record array
        rec_array = rec['record']
        if rec_array.dtype.names is None:
            # Empty record -- skip
            continue

        if 'magnitude' not in rec_array.dtype.names:
            logger.error('Cannot map magnitude of change')
            logger.error('Version of result file: {v}'.format(
                v=rec['version'] if 'version' in rec.files else 'Unknown'))
            raise KeyError('Magnitude information not present in file %s -- '
                           'has it been calculated?' % result)

        changed = np.where(rec_array['break'] != 0)[0]
        if changed.size == 0:
            continue

        logger.debug('Using non-zero elements of "magnitude" field in '
                     'changed records for magnitude indices')
        return np.nonzero(np.any(rec_array[changed]['magnitude'] != 0))[0]


# MAPPING FUNCTIONS
def get_change_date(start, end, result_location, image_ds,
                    first=False,
                    out_format='%Y%j',
                    magnitude=False,
                    ndv=-9999, pattern='yatsm_r*', warn_on_empty=False):
    """ Output raster with changemap

    Args:
        start (int): Ordinal date for start of map records
        end (int): Ordinal date for end of map records
        result_location (str): Location of results
        image_ds (gdal.Dataset): Example dataset
        first (bool): Use first change instead of last
        out_format (str, optional): Output date format
        magnitude (bool, optional): output magnitude of each change?
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results
        warn_on_empty (bool, optional): Log warning if result contained no
            result records (default: False)


    Returns:
        tuple: A 2D np.ndarray array containing the changes between the
            start and end date. Also includes, if specified, a 3D np.ndarray of
            the magnitude of each change plus the indices of these magnitudes

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    datemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.int32) * int(ndv)
    # Determine what magnitude information to output if requested
    if magnitude:
        magnitude_indices = get_magnitude_indices(records)
        magnitudemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize,
                                magnitude_indices.size),
                               dtype=np.float32) * float(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=warn_on_empty):

        index = np.where((rec['break'] >= start) &
                         (rec['break'] <= end))[0]

        if first:
            _, _index = np.unique(rec['px'][index], return_index=True)
            index = index[_index]

        if index.shape[0] != 0:
            if out_format != 'ordinal':
                dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                  for _d in rec['break'][index]])
                datemap[rec['py'][index], rec['px'][index]] = dates
            else:
                datemap[rec['py'][index], rec['px'][index]] = \
                    rec['break'][index]
            if magnitude:
                magnitudemap[rec['py'][index], rec['px'][index], :] = \
                    rec[index]['magnitude'][:, magnitude_indices]

    if magnitude:
        return datemap, magnitudemap, magnitude_indices
    else:
        return datemap, None, None


def get_change_num(start, end, result_location, image_ds,
                   ndv=-9999, pattern='yatsm_r*', warn_on_empty=False):
    """ Output raster with changemap

    Args:
        start (int): Ordinal date for start of map records
        end (int): Ordinal date for end of map records
        result_location (str): Location of results
        image_ds (gdal.Dataset): Example dataset
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results
        warn_on_empty (bool, optional): Log warning if result contained no
            result records (default: False)

    Returns:
        np.ndarray: 2D numpy array containing the number of changes
            between the start and end date; list containing band names

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=warn_on_empty):
        # X location of each changed model
        px_changed = rec['px'][(rec['break'] >= start) & (rec['break'] <= end)]
        # Count occurrences of changed pixel locations
        bincount = np.bincount(px_changed)
        # How many changes for unique values of px_changed?
        n_change = bincount[np.nonzero(bincount)[0]]

        # Add in the values
        px = np.unique(px_changed)
        py = rec['py'][np.in1d(px, rec['px'])]
        raster[py, px] = n_change

    return raster
