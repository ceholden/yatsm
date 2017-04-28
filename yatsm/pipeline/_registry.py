""" Find and register tasks available for pipeline
"""
import logging
from pkg_resources import iter_entry_points

logger = logging.getLogger(__name__)


TASKS_ENTRY_POINT = 'yatsm.pipeline.tasks'


def _get_ep(ep):
    d = {}
    for _ep in iter_entry_points(ep):
        try:
            d[_ep.name] = _ep.load()
        except Exception as e:
            logger.exception('Could not load pipeline task: {}'
                             .format(_ep.name), e)
    return d


def find_tasks():
    """ Return pipeline tasks

    Returns:
        dict: A dict mapping task name to task function for all tasks
        discovered
    """
    # Imports for "known" or "included" tasks are put inside the function
    # to prevent circular import. Otherwise, pipeline.__init__ imports this,
    # which import tasks.___, each of which require pipeline._validation
    from yatsm import tasks

    pipeline_tasks = {
        # CHANGE
        'pixel_CCDCesque': tasks.pixel_CCDCesque,
        # IO
        'xarray_open': tasks.xarray_open,
        # STASH
        'sklearn_load': tasks.sklearn_load,
        'sklearn_dump': tasks.sklearn_dump,
        # DATA MANIPULATION
        'dmatrix': tasks.dmatrix,
        'norm_diff': tasks.norm_diff
    }
    pipeline_tasks.update(_get_ep(TASKS_ENTRY_POINT))

    return pipeline_tasks
