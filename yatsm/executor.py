""" Handler for execution using dask or dask.distributed
"""
from functools import partial
import logging

logger = logging.getLogger(__name__)

EXECUTOR_TYPES = ['sync', 'thread', 'process', 'distributed']

EXECUTOR_DEFAULTS = {
    'sync': None,
    'thread': 1,
    'process': 1,
    'distributed': '127.0.0.1:8786'
}


def _distributed_executor(executor, args):
    try:
        import distributed
    except ImportError:
        logger.critical('You must have "distributed" installed')
        raise


    class DistributedExecutor(object):
        """ concurrent.futures-like dask.distributed executor
        """
        def __init__(self, executor):
            self._executor = executor

        def submit(self, func, *args, **kwargs):
            return self._executor.submit(func, *args, **kwargs)

        @staticmethod
        def as_completed(futures):
            return distributed.as_completed(futures)

    return DistributedExecutor(distributed.Client(str(args)))


def _futures_executor(executor, args):
    try:
        from concurrent.futures import (ProcessPoolExecutor, ThreadPoolExecutor,
                                        as_completed)
    except ImportError as err:
        logger.critical('You must have Python3 or "futures" package installed.')
        raise

    class FuturesExecutor(object):
        def __init__(self, executor):
            self._executor = executor

        def submit(self, func, *args, **kwargs):
            return self._executor.submit(func, *args, **kwargs)

        @staticmethod
        def as_completed(futures):
            return as_completed(futures)

    n = int(args) if args else 1
    return FuturesExecutor(ProcessPoolExecutor(n) if executor == 'process' else
                           ThreadPoolExecutor(n))


def get_executor(executor, arg):
    """ Click callback for determining executor type
    """
    try:
        exc = (_distributed_executor(executor, arg) if executor == 'distributed'
               else _futures_executor(executor, arg))
    except Exception as err:
        logger.exception('Could not setup an executor of type "{}" '
                         'with args "{}": "{}"'
                         .format(executor, arg, err))
        raise
    else:
        return exc
