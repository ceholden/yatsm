""" Various datetime tools
"""
import datetime as dt
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def datetime2int(data, out_format, in_format=None, dtype=np.int32):
    """ Return input data in an integer friendly date format

    Args:
        data (np.ndarray): Time data
        out_format (str): Either 'ordinal' for ordinal data (see
            :ref:`dt.datetime.toordinal), or a date string (e.g.,
            '%Y%m%d')
        in_format (str): Optionally, if `data` is a character dtype
            then provide the date format for it
        dtype (np.dtype): Output datatype

    Returns:
        np.ndarray: Integer date representation
    """
    if data.dtype.kind == 'i':
        logger.debug('Assuming ordinal data')
        _data = pd.Series(data).map(dt.datetime.fromordinal)
    elif dtype.dtype.type is np.datetime64:
        logger.debug('Assuming datetime-like')
        _data = pd.Series(data)

    # common among all steps
    if out_format == 'ordinal':
        return _data.map(dt.datetime.toordinal).astype(dtype).values
    else:
        return _data.dt.strftime(out_format).astype(dtype).values
