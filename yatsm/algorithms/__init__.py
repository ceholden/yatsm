""" Submodule for YATSM algorithms

Algorithms currently include:
    - :py:class:`ccdc.CCDCesque`

"""
from collections import defaultdict
import functools
from pkg_resources import iter_entry_points

from .ccdc import CCDCesque


def _broken_ep(ep, exc, *args, **kwargs):
    """ Delay error due to broken entry point until it executes
    """
    import logging
    logger = logging.getLogger('yatsm')
    logger.critical('Trying to import "{0.name}" algorithm entry point '
                    'raised a {1}'.format(ep, exc))
    raise exc


def _iep(s):
    """ Handle loading entry point 's' through iter entry points
    """
    d = defaultdict(dict)
    for _ep in iter_entry_points(s):
        try:
            d[_ep.name] = _ep.load()
        except Exception as e:
            d[_ep.name] = functools.partial(_broken_ep, _ep, e)
    return d


available = {
    'change': _iep('yatsm.algorithms.change'),
    'postprocess': _iep('yatsm.algorithms.postprocess')  # TODO: use this ep!
}
