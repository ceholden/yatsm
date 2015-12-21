""" Functions relevant for mapping phenology fit information
"""
import logging

import numpy as np

from .utils import find_indices
from ..utils import find_results, iter_records

logger = logging.getLogger('yatsm')


def get_phenology(date, result_location, image_ds,
                  after=False, before=False, qa=False,
                  ndv=-9999, pattern='yatsm_r*',
                  warn_on_empty=False):
    """ Output a raster containing phenology information

    Phenology information includes spring_doy, autumn_doy, pheno_cor, peak_evi,
    peak_doy, and pheno_nobs.

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before)
        ndv (int, optional): NoDataValue
        pattern (str, optional): filename pattern of saved record results
        warn_on_empty (bool, optional): Log warning if result contained no
            result records (default: False)

    Returns:
        tuple: A tuple (np.ndarray, list) containing the 3D np.ndarray of the
            phenology metrics, and the band names for the output dataset

    """
    # Find results
    records = find_results(result_location, pattern)

    n_bands = 7
    attributes = ['spring_doy', 'autumn_doy', 'pheno_cor', 'peak_evi',
                  'peak_doy', 'pheno_nobs']
    band_names = ['SpringDOY', 'AutumnDOY', 'Pheno_R*10000', 'Peak_EVI*10000',
                  'Peak_DOY', 'Pheno_NObs', 'GrowingDOY']
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')

    logger.debug('Allocating memory')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=warn_on_empty):
        if not all([_attr in rec.dtype.names for _attr in attributes]):
            raise ValueError('Results do not have phenology metrics')

        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            # Apply scale factors for R and peak EVI
            rec['pheno_cor'][index] *= 10000.0
            rec['peak_evi'][index] *= 10000.0

            for _b, _attr in enumerate(attributes):
                raster[rec['py'][index],
                       rec['px'][index], _b] = rec[_attr][index]
            raster[rec['py'][index],
                   rec['px'][index], 6] = \
                rec['autumn_doy'][index] - rec['spring_doy'][index]
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names
