""" Functions for running various processing task in a pipeline

.. todo::

    Allow specification of pipeline tasks using entry points

"""
import logging
from pkg_resources import iter_entry_points

from ._validation import eager_task, outputs, requires, version
from .change import pixel_CCDCesque
from .preprocess import dmatrix, norm_diff
from .stash import sklearn_dump, sklearn_load

logger = logging.getLogger(__name__)


def _get_eps(ep):
    d = {}
    for _ep in iter_entry_points(ep):
        try:
            d[_ep.name] = _ep.load()
        except Exception as e:
            logger.exception('Could not load pipeline task: {}'
                             .format(_ep.name), e)
    return d


#: dict: Tasks that generate segments, usually through
#        some kind of change detection process
SEGMENT_TASKS = {
    # CHANGE
    'pixel_CCDCesque': pixel_CCDCesque,
}


PIPELINE_TASKS = {
    # STASH
    'sklearn_load': sklearn_load,
    'sklearn_dump': sklearn_dump,
    # DATA MANIPULATION
    'dmatrix': dmatrix,
    'norm_diff': norm_diff
}


SEGMENT_TASKS.update(_get_eps('yatsm.pipeline.tasks.segment'))
PIPELINE_TASKS.update(_get_eps('yatsm.pipeline.tasks.tasks'))
PIPELINE_TASKS.update(SEGMENT_TASKS)


__all__ = [
    'eager_task',
    'outputs',
    'requires',
    'version'
]
