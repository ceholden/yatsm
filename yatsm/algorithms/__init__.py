""" Submodule for YATSM algorithms

Algorithms currently include:
    - :py:class:`ccdc.CCDCesque`

"""
from ._core import iep, Segment, SEGMENT_ATTRS, SEGMENT_DTYPES
from .ccdc import CCDCesque


AVAILABLE = {
    'change': iep('yatsm.algorithms.change'),
    'postprocess': iep('yatsm.algorithms.postprocess')  # TODO: use this ep!
}


__all__ = [
    'Segment'
]
