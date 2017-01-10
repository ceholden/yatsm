""" Core datatypes for YATSM compatible estimators
"""
from collections import defaultdict
import functools
from pkg_resources import iter_entry_points

import numpy as np


# Segment related
#: list: Datatype for segment useful for NumPy structurred arrays
SEGMENT_DTYPES = [
    ('start_day', 'i4'),
    ('end_day', 'i4'),
    ('break_day', 'i4'),
    ('px', 'float'),
    ('py', 'float')
]


#: list: Names of segment attributes
SEGMENT_ATTRS = [name for name, dtype in SEGMENT_DTYPES]


class Segment(np.ndarray):
    """ YATSM Time Series Segment as NumPy Structured Array
    """

    @classmethod
    def from_example(cls, **attrs):
        """ Return a Segment from example name=value pairs

        Args:
            attrs (dict): Collection of Segment attributes to
                use in Segment definition

        Returns:
            Segment: An empty structured :ref:`np.ndarray` with data types
            for default segment attributes and any others listed in ``attrs``
        """
        dtypes = list(SEGMENT_DTYPES)
        for name, arr in attrs.items():
            shp = getattr(arr, 'shape', 0)
            dtype = getattr(arr, 'dtype', 'f4')

            dtypes.append((name, dtype, shp) if shp else (name, dtype))

        return cls(shape=1, dtype=dtypes)

    @classmethod
    def barebones(cls, **kwds):
        """ Return a barebones, minimum required fields, Segment
        """
        return cls(shape=1, dtype=SEGMENT_DTYPES)


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
