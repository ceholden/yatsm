""" Functions relevant for mapping statistical model predictions or fits
"""
import logging
import re

import numpy as np
import patsy

from .utils import find_result_attributes, find_indices
from ..utils import find_results, iter_records
from ..regression.transforms import harm

logger = logging.getLogger('yatsm')


def get_coefficients(date, result_location, image_ds,
                     bands, coefs,
                     prefix='', amplitude=False,
                     after=False, before=False, qa=False,
                     ndv=-9999, pattern='yatsm_r*', warn_on_empty=False):
    """ Output a raster with coefficients from CCDC

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        bands (list): Bands to predict
        coefs (list): List of coefficients to output
        image_ds (gdal.Dataset): Example dataset
        prefix (str, optional): Use coef/rmse with refit prefix (default: '')
        amplitude (bool, optional): Map amplitude of seasonality instead of
            individual coefficient estimates for sin/cosine pair
            (default: False)
        after (bool, optional): If date intersects a disturbed period, use next
            available time segment (default: False)
        before (bool, optional): If date does not intersect a model, use
            previous non-disturbed time segment (default: False)
        qa (bool, optional): Add QA flag specifying segment type (intersect,
            after, or before) (default: False)
        ndv (int, optional): NoDataValue (default: -9999)
        pattern (str, optional): filename pattern of saved record results
        warn_on_empty (bool, optional): Log warning if result contained no
            result records (default: False)

    Returns:
        tuple: A tuple (np.ndarray, list) containing the 3D numpy.ndarray of
            the coefficients (coefficient x band x pixel), and the band names
            for the output dataset

    """
    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, i_coefs, use_rmse, coef_names, _, _ = find_result_attributes(
        records, bands, coefs, prefix=prefix)

    # Process amplitude transform for seasonality coefficients
    if amplitude:
        harm_coefs = []
        for i, (c, n) in enumerate(zip(i_coefs, coef_names)):
            if re.match(r'harm\(x, [0-9]+\)\[0]', n):
                harm_coefs.append(c)
                coef_names[i] = re.sub(r'harm(.*)\[.*', r'amplitude\1', n)
        # Remove sin term from each harmonic pair
        i_coefs = [c for c in i_coefs if c - 1 not in harm_coefs]
        coef_names = [n for n in coef_names if 'harm' not in n]

    n_bands = len(i_bands)
    n_coefs = len(i_coefs)
    n_rmse = n_bands if use_rmse else 0

    # Setup output band names
    band_names = []
    for _c in coef_names:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_' + _c.replace(' ', ''))
    if use_rmse is True:
        for _b in i_bands:
            band_names.append('B' + str(_b + 1) + '_RMSE')
    n_qa = 0
    if qa:
        n_qa += 1
        band_names.append('SegmentQAQC')
    n_out_bands = n_bands * n_coefs + n_rmse + n_qa

    _coef = prefix + 'coef' if prefix else 'coef'
    _rmse = prefix + 'rmse' if prefix else 'rmse'

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_out_bands),
                     dtype=np.float32) * ndv

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=warn_on_empty):
        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            if n_coefs > 0:
                # Normalize intercept to mid-point in time segment
                rec[_coef][index, 0, :] += (
                    (rec['start'][index] + rec['end'][index]) /
                    2.0)[:, np.newaxis] * \
                    rec[_coef][index, 1, :]

                # If we want amplitude, calculate it
                if amplitude:
                    for harm_coef in harm_coefs:
                        rec[_coef][index, harm_coef, :] = np.linalg.norm(
                            rec[_coef][index, harm_coef:harm_coef + 2, :],
                            axis=1)

                # Extract coefficients
                raster[rec['py'][index],
                       rec['px'][index], :n_coefs * n_bands] =\
                    np.reshape(rec[_coef][index][:, i_coefs, :][:, :, i_bands],
                               (index.size, n_coefs * n_bands))

            if use_rmse:
                raster[rec['py'][index], rec['px'][index],
                       n_coefs * n_bands:n_out_bands - n_qa] =\
                    rec[_rmse][index][:, i_bands]
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names


def get_prediction(date, result_location, image_ds,
                   bands='all', prefix='',
                   after=False, before=False, qa=False,
                   ndv=-9999, pattern='yatsm_r*', warn_on_empty=False):
    """ Output a raster with the predictions from model fit for a given date

    Args:
        date (int): Ordinal date for prediction image
        result_location (str): Location of the results
        image_ds (gdal.Dataset): Example dataset
        bands (str, list): Bands to predict - 'all' for every band, or specify
            a list of bands
        prefix (str, optional): Use coef/rmse with refit prefix (default: '')
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
        np.ndarray: A 3D numpy.ndarray containing the prediction for each band,
            for each pixel

    """
    # Find results
    records = find_results(result_location, pattern)

    # Find result attributes to extract
    i_bands, _, _, _, design, design_info = find_result_attributes(
        records, bands, None, prefix=prefix)

    n_bands = len(i_bands)
    band_names = ['Band_{0}'.format(b) for b in range(n_bands)]
    if qa:
        n_bands += 1
        band_names.append('SegmentQAQC')
    n_i_bands = len(i_bands)

    # Create X matrix from date -- ignoring categorical variables
    if re.match(r'.*C\(.*\).*', design):
        logger.warning('Categorical variable found in design matrix not used'
                       ' in predicted image estimate')
    design = re.sub(r'[\+\-][\ ]+C\(.*\)', '', design)
    X = patsy.dmatrix(design, {'x': date}).squeeze()

    i_coef = []
    for k, v in design_info.items():
        if not re.match('C\(.*\)', k):
            i_coef.append(v)
    i_coef = np.sort(i_coef)

    logger.debug('Allocating memory')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize, n_bands),
                     dtype=np.int16) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records, warn_on_empty=warn_on_empty):
        for _qa, index in find_indices(rec, date, after=after, before=before):
            if index.shape[0] == 0:
                continue

            # Calculate prediction
            _coef = rec['coef'].take(index, axis=0).\
                take(i_coef, axis=1).take(i_bands, axis=2)
            raster[rec['py'][index], rec['px'][index], :n_i_bands] = \
                np.tensordot(_coef, X, axes=(1, 0))
            if qa:
                raster[rec['py'][index], rec['px'][index], -1] = _qa

    return raster, band_names
