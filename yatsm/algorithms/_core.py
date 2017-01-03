""" Core datatypes for YATSM compatible estimators
"""
from collections import defaultdict
import functools
from pkg_resources import iter_entry_points

import numpy as np

# Segment related
#: list: Datatype for segment useful for NumPy structurred arrays
SEGMENT_DTYPE = [
    ('start_day', 'i4'),
    ('end_day', 'i4'),
    ('break_day', 'i4'),
    ('px', 'float'),
    ('py', 'float')
]


#: list: Names of segment attributes
SEGMENT_ATTRS = [name for name, dtype in SEGMENT_DTYPE]


# Entry point handling
def broken_ep(ep, exc, *args, **kwargs):
    """ Delay error due to broken entry point until it executes
    """
    import logging
    logger = logging.getLogger('yatsm')
    logger.critical('Trying to import "{0.name}" algorithm entry point '
                    'raised a {1}'.format(ep, exc))
    raise exc


def iep(s):
    """ Handle loading entry point 's' through iter entry points
    """
    d = defaultdict(dict)
    for _ep in iter_entry_points(s):
        try:
            d[_ep.name] = _ep.load()
        except Exception as e:
            d[_ep.name] = functools.partial(broken_ep, _ep, e)
    return d


