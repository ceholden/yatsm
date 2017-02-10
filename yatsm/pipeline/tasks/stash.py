""" Tasks that return Python objects used by other tasks
"""
import logging

from sklearn.externals import joblib as jl

from yatsm.pipeline.tasks._validation import (eager_task, requires, outputs,
                                              version)

logger = logging.getLogger(__name__)


@version('sklearn_load:1.0.0')
@eager_task
@outputs(stash=[str])
def sklearn_load(pipe, require, output, config=None):
    """ Load a scikit-learn estimator

    Args:
        filename (str): Filename to read from

    Returns:
        sklearn.base.BaseEstimator: Estimator
    """
    pipe.cache[output['cache'][0]] = jl.load(config['filename'])
    return pipe


@version('sklearn_load:1.0.0')
@eager_task
@requires(stash=[str])
def sklearn_dump(pipe, require, output, config=None):
    """ Saves a scikit-learn estimator using ``joblib``

    Args:
        filename (str): Filename to write to

    """
    jl.dump(config['filename'], pipe.cache[output['cache'][0]],
            compress=config.get('compress', 3))
    return pipe
