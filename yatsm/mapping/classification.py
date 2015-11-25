""" Functions relevant for mapping categorical classification labels
"""
import logging

import numpy as np

from .utils import find_indices
from ..utils import find_results, iter_records

logger = logging.getLogger('yatsm')


def get_classification(date, result_location, image_ds,
                       after=False, before=False, qa=False,
                       pred_proba=False,
                       ndv=0, pattern='yatsm_r*', warn_on_empty=False):
    """ Output raster with classification results

    Args:
        date (int): ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before)
        pred_proba (bool, optional): Include additional band with
            classification value probabilities
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results
        warn_on_empty (bool, optional): Log warning if result contained no
            result records (default: False)

    Returns:
        np.ndarray: 2D numpy array containing the classification map for the
            date specified

    """
    # Find results
    records = find_results(result_location, pattern)

    n_bands = 2 if pred_proba else 1
    dtype = np.uint16 if pred_proba else np.uint8

    band_names = ['Classification']
    if pred_proba:
        band_names.append('Pred Proba (x10,000)')
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=dtype) * int(ndv)

    logger.debug('Processing results')
    for rec, fname in iter_records(records, warn_on_empty=warn_on_empty,
                                   yield_filename=True):
        if 'class' not in rec.dtype.names:
            logger.warning('Results in {f} do not have classification labels'
                           .format(f=fname))
            continue
        if 'class_proba' not in rec.dtype.names and pred_proba:
            raise ValueError('Results do not have classification prediction'
                             ' probability values')

        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            raster[rec['py'][index],
                   rec['px'][index], 0] = rec['class'][index]
            if pred_proba:
                raster[rec['py'][index],
                       rec['px'][index], 1] = \
                    rec['class_proba'][index].max(axis=1) * 10000
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names
