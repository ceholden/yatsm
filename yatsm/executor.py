""" Handler for execution using dask or dask.distributed
"""
import logging
import sys

PY2 = sys.version_info[0] == 2

HAS_CONCURRENT = True
HAS_DISTRIBUTED = True

try:
    from concurrent.futures import (Future,
                                    ProcessPoolExecutor,
                                    ThreadPoolExecutor,
                                    as_completed)
except ImportError:
    HAS_CONCURRENT = False
try:
    import distributed
except ImportError:
    HAS_DISTRIBUTED = False

logger = logging.getLogger(__name__)

EXECUTOR_TYPES = ['sync', 'thread', 'process', 'distributed']

EXECUTOR_DEFAULTS = {
    'sync': None,
    'thread': 1,
    'process': 1,
    'distributed': '127.0.0.1:8786'
}


class _Executor(object):

    def submit(self, func, *args, **kwds):
        raise NotImplementedError('Subclass should do this')

    @staticmethod
    def _result(future):
        raise NotImplementedError('Subclass should do this')

    @staticmethod
    def as_completed(futures):
        raise NotImplementedError('Subclass should do this')

    def shutdown(self, timeout=10, futures=None):
        raise NotImplementedError('Subclass should do this')


class SyncExecutor(_Executor):
    """ :mod:`concurrent.futures` executor wrapper
    """

    def submit(self, func, *args, **kwds):
        future = Future()
        future._spec = (func, args, kwds)
        return future

    @staticmethod
    def _result(future):
        func, args, kwds = future._spec
        try:
            result = func(*args, **kwds)
        except Exception as e:
            if PY2:
                tb = sys.exc_info()
                future.set_exception_info(e, tb[2])
            else:
                future.set_exception(e)
        else:
            future.set_result(result)
        return future

    @staticmethod
    def as_completed(futures):
        for future in futures:
            yield SyncExecutor._result(future)

    def shutdown(self, timeout=10, futures=None):
        return


class ConcurrentExecutor(_Executor):
    """ :mod:`concurrent.futures` executor wrapper
    """
    def __init__(self, executor):
        self._executor = executor

    def submit(self, func, *args, **kwds):
        return self._executor.submit(func, *args, **kwds)

    @staticmethod
    def as_completed(futures):
        return as_completed(futures)

    def shutdown(self, timeout=10, futures=None):
        self._executor.shutdown(wait=timeout > 0)


class DistributedExecutor(_Executor):
    """ :mod:`concurrent.futures`-like dask.distributed executor
    """
    def __init__(self, executor):
        self._executor = executor

    def submit(self, func, *args, **kwds):
        return self._executor.submit(func, *args, **kwds)

    @staticmethod
    def as_completed(futures):
        return distributed.as_completed(futures)

    def shutdown(self, timeout=10, futures=None):
        self._executor.cancel(futures)


def _sync_executor(executor, arg):
    if not HAS_CONCURRENT:
        raise ImportError('You must have Python3 or "futures" package '
                          'installed to use SyncExecutor.')
    return SyncExecutor()


def _concurrent_executor(executor, args):
    if not HAS_CONCURRENT:
        raise ImportError('You must have Python3 or "futures" package '
                          'installed to use ConcurrentExecutor.')
    n = int(args) if args else 1
    return ConcurrentExecutor(ProcessPoolExecutor(n) if executor == 'process'
                              else ThreadPoolExecutor(n))


def _distributed_executor(executor, args):
    if not HAS_DISTRIBUTED:
        raise ImportError('You must have "distributed" installed to use '
                          'DistributedExecutor')

    return DistributedExecutor(distributed.Client(str(args)))


def get_executor(executor, arg):
    """ Click callback for determining executor type
    """
    _map = {
        'sync': _sync_executor,
        'thread': _concurrent_executor,
        'process': _concurrent_executor,
        'distributed': _distributed_executor
    }
    if executor not in _map:
        raise KeyError("Unknown executor '{}'".format(executor))
    try:
        exc = _map[executor](executor, arg)
    except Exception as err:
        logger.exception('Could not setup an executor of type "{}" '
                         'with args "{}": "{}"'
                         .format(executor, arg, err))
        raise
    else:
        return exc
