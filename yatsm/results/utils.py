""" Result helper utility functions
"""
import logging
import os
import re

import rasterio.windows

logger = logging.getLogger(__name__)

RESULT_TEMPLATE = (
    'yatsm_'
    'r{row_off:04d}_{num_rows:04d}'
    '_c{col_off:04d}_{num_cols:04d}.h5'
)
RESULT_ATTRS = rasterio.windows.Window._fields


def result_filename(window, root='.', pattern=RESULT_TEMPLATE):
    """ Return filename for a result from a root dir and pattern

    Attributes available for filename pattern filling with ``.format`` are
    ones available from ``window.todict()``, including::

        * `col_off`
        * `row_off`
        * `num_cols`
        * `num_rows`

    Args:
        window (rasterio.windows.Window): Optionally, a :ref:`rasterio.Window`
            tuple to use for information
        root (str): Root directory
        pattern (str): Format string

    Returns:
        str: Filename filled out
    """
    if window and type(window) is tuple:
        window = rasterio.windows.Window.from_ranges(*window)

    try:
        out = os.path.join(root, pattern.format(**window.todict()))
    except KeyError as ke:
        logger.exception('Unknown attribute used for output result pattern. '
                         'Potential choices include: {}'
                         .format(', '.join(RESULT_ATTRS)))
        raise
    else:
        return out


def pattern_to_regex(pattern):
    """ Invert a result filename pattern and return regex search expression
    """
    def _repl(match):
        s = match.group()
        if s[-2] == 'd':
            return '[0-9]*'
        else:
            return '*'
    search = re.sub('{.*?}', _repl, pattern)
    return search
