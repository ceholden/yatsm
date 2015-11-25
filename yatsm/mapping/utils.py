""" Utilities for turning YATSM record results into maps

Also stores definitions for model QA/QC values
"""
import logging

import numpy as np

from ..regression import design_to_indices

logger = logging.getLogger('yatsm')

# QA/QC values for segment types
MODEL_QA_QC = {
    'INTERSECT': 3,
    'AFTER': 2,
    'BEFORE': 1
}


def find_result_attributes(results, bands, coefs, prefix=''):
    """ Returns attributes about the dataset from result files

    Args:
        results (list): Result filenames
        bands (list): Bands to describe for output
        coefs (list): Coefficients to describe for output
        prefix (str, optional): Search for coef/rmse results with given prefix
            (default: '')

    Returns:
        tuple: Tuple containing `list` of indices for output bands and output
            coefficients, `bool` for outputting RMSE, `list` of coefficient
            names, `str` design specification, and `OrderedDict` design_info
            (i_bands, i_coefs, use_rmse, design, design_info)

    Raises:
        KeyError: Raise KeyError when a required result output is missing
            from the saved record structure

    """
    _coef = prefix + 'coef' if prefix else 'coef'
    _rmse = prefix + 'rmse' if prefix else 'rmse'

    # How many coefficients and bands exist in the results?
    n_bands, n_coefs = None, None
    design = None
    for r in results:
        try:
            _result = np.load(r)
            rec = _result['record']
            # Handle pre/post v0.5.4 (see issue #53)
            if 'metadata' in _result.files:
                logger.debug('Finding X design info for version>=v0.5.4')
                md = _result['metadata'].item()
                design = md['YATSM']['design']
                design_str = md['YATSM']['design_matrix']
            else:
                logger.debug('Finding X design info for version<0.5.4')
                design = _result['design_matrix'].item()
                design_str = _result['design'].item()
        except:
            continue

        if not rec.dtype.names:
            continue

        if _coef not in rec.dtype.names or _rmse not in rec.dtype.names:
            if prefix:
                logger.error('Coefficients and RMSE not found with prefix %s. '
                             'Did you calculate them?' % prefix)
            raise KeyError('Could not find coefficients ({0}) and RMSE ({1}) '
                           'in record'.format(_coef, _rmse))

        try:
            n_coefs, n_bands = rec[_coef][0].shape
        except:
            continue
        else:
            break

    if n_coefs is None:
        raise KeyError('Could not determine the number of coefficients')
    if n_bands is None:
        raise KeyError('Could not determine the number of bands')
    if design is None:
        raise KeyError('Design matrix specification not found in results')

    # How many bands does the user want?
    if bands == 'all':
        i_bands = range(0, n_bands)
    else:
        # NumPy index on 0; GDAL on 1 -- so subtract 1
        i_bands = [b - 1 for b in bands]
        if any([b > n_bands for b in i_bands]):
            raise KeyError('Bands specified exceed size of bands in results')

    # How many coefficients did the user want?
    use_rmse = False
    if coefs:
        if 'rmse' in coefs or 'all' in coefs:
            use_rmse = True
        i_coefs, coef_names = design_to_indices(design, coefs)
    else:
        i_coefs, coef_names = None, None

    logger.debug('Bands: {0}'.format(i_bands))
    if coefs:
        logger.debug('Coefficients: {0}'.format(i_coefs))

    return (i_bands, i_coefs, use_rmse, coef_names, design_str, design)


def find_indices(record, date, after=False, before=False):
    """ Yield indices matching time segments for a given date

    Args:
      record (np.ndarray): Saved model result
      date (int): Ordinal date to use when finding matching segments
      after (bool, optional): If date intersects a disturbed period, use next
        available time segment
      before (bool, optional): If date does not intersect a model, use previous
        non-disturbed time segment

    Yields:
      tuple: (int, np.ndarray) the QA value and indices of `record` containing
        indices matching criteria. If `before` or `after` are specified,
        indices will be yielded in order of least desirability to allow
        overwriting -- `before` indices, `after` indices, and intersecting
        indices.

    """
    if before:
        # Model before, as long as it didn't change
        index = np.where((record['end'] <= date) & (record['break'] == 0))[0]
        yield MODEL_QA_QC['BEFORE'], index

    if after:
        # First model starting after date specified
        index = np.where(record['start'] >= date)[0]
        _, _index = np.unique(record['px'][index], return_index=True)
        index = index[_index]
        yield MODEL_QA_QC['AFTER'], index

    # Model intersecting date
    index = np.where((record['start'] <= date) & (record['end'] >= date))[0]
    yield MODEL_QA_QC['INTERSECT'], index
