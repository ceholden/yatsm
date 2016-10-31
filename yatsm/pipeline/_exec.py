""" Functions for handling the execution of a pipeline graph
"""
import logging

from dask import delayed

logger = logging.getLogger(__name__)


def delay_pipeline(pipeline, pipe):
    """ Return a ``dask.delayed`` pipeline ready to execute

    Args:
        pipeline (list[Task]): A list of curried ``Task`` ready to be
            run using data from ``pipe``. This list may be constructed as the
            output of :ref:`setup_pipeline`, for example.
        pipe (dict): Dictionary storing ``data`` and ``record`` information.

    Returns:
        dask.delayed: A delayed pipeline ready to be executed
    """
    _pipeline = delayed(pipeline[0].curry())(pipe)
    for task in pipeline[1:]:
        _pipeline = delayed(task.curry())(_pipeline)

    return _pipeline
