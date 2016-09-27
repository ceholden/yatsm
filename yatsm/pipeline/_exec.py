""" Functions for handling the execution of a pipeline graph
"""
import logging

from toolz import curry
from dask import delayed

from ._topology import config_to_tasks
from .language import OUTPUT, REQUIRE
from .tasks import PIPELINE_TASKS

logger = logging.getLogger(__name__)


def curry_pipeline_task(func, spec):
    return curry(func,
                 **{REQUIRE: spec[REQUIRE],
                    OUTPUT: spec[OUTPUT],
                    'config': spec.get('config', {})})


def setup_pipeline(config, pipe, overwrite=True):
    """ Process the configuration for a YATSM pipeline

    Args:
        config (dict): Pipeline configuration dictionary
        pipe (dict[str: dict]): Dictionary storing ``data`` and ``record``
            information. At this point, ``data`` and ``record`` can either
            store actual data (e.g., an `xarray.Dataset`) or simply a
            dictionary that mimics the data (i.e., it contains the same keys).
        overwrite (bool): Allow tasks to overwrite values that have already
            been computed

    Returns:
        tuple(list, list): Two lists of curried functions ready to be ran in a
            pipeline. The first list contains functions that are "eager",
            meaning that they may be ran for all pixels in the dataset
    """
    tasks = config_to_tasks(config, pipe, overwrite=overwrite)

    halt_eager, eager_pipeline = False, []
    pipeline = []
    for task in tasks:
        try:
            func = PIPELINE_TASKS[config[task]['task']]
        except KeyError as exc:
            msg = 'Unknown pipeline task "{}" referenced in "{}"'.format(
                config[task]['task'], task)
            logger.error(msg)
            raise KeyError(msg)

        is_eager = getattr(func, 'is_eager', False)
        if is_eager and not halt_eager:
            eager_pipeline.append(curry_pipeline_task(func, config[task]))
        else:
            if is_eager:
                logger.debug('Not able to compute eager function "{}" on all '
                             'all pixels at once because it can after '
                             'non-eager tasks'.format(task))
            pipeline.append(curry_pipeline_task(func, config[task]))
            halt_eager = True

    return eager_pipeline, pipeline


def delay_pipeline(pipeline, pipe):
    """ Return a ``dask.delayed`` pipeline ready to execute

    Args:
        pipeline (list[callable]): A list of curried functions ready to be
            run using data from ``pipe``. This list may be constructed as the
            output of :ref:`setup_pipeline`, for example.
        pipe (dict): Dictionary storing ``data`` and ``record`` information.

    Returns:
        dask.delayed: A delayed pipeline ready to be executed
    """
    _pipeline = delayed(pipeline[0])(pipe)
    for task in pipeline[1:]:
        _pipeline = delayed(task)(_pipeline)

    return _pipeline
