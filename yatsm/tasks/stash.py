""" Tasks that return Python objects used by other tasks
"""
import logging

from sklearn.externals import joblib as jl

from yatsm.pipeline import (
    language,
    eager_task, task_version,
    outputs, requires
)

logger = logging.getLogger(__name__)


@task_version('sklearn_load:1.0.0')
@eager_task
@outputs(stash=[str])
def sklearn_load(pipe, require, output, config=None):
    """ Load a scikit-learn estimator

    Args:
        filename (str): Filename to read from

    Returns:
        sklearn.base.BaseEstimator: Estimator
    """
    pipe.cache[output[language.STASH][0]] = jl.load(config['filename'])
    return pipe


@task_version('sklearn_load:1.0.0')
@eager_task
@requires(stash=[str])
def sklearn_dump(pipe, require, output, config=None):
    """ Saves a scikit-learn estimator using ``joblib``

    Args:
        filename (str): Filename to write to

    """
    jl.dump(config['filename'],
            pipe.stash[output[language.STASH][0]],
            compress=config.get('compress', 3))
    return pipe
