""" Handler for execution using dask or dask.distributed
"""
from functools import partial
import logging

logger = logging.getLogger(__name__)

EXECUTOR_TYPES = ['sync', 'thread', 'multiprocess', 'distributed']

EXECUTOR_DEFAULTS = {
    'sync': None,
    'thread': 1,
    'multiprocess': 1,
    'distributed': '127.0.0.1:8786'
}


def _parse_num_workers(value):
    return int(value) if value else 1


def get_executor(exc_type, arg):
    """ Click callback for determining executor type
    """
    import dask

    def _get_executor(exc_type, arg):
        if exc_type == 'thread':
            return partial(dask.threaded.get,
                           num_workers=_parse_num_workers(arg))
        elif exc_type == 'multiprocess':
            return partial(dask.multiprocessing.get,
                           num_workers=_parse_num_workers(arg))
        elif exc_type == 'distributed':
            import distributed
            client = distributed.Client(arg)
            return client.get
        elif exc_type == 'sync':
            return dask.async.get_sync
        else:
            raise KeyError('Unknown executor type "{}"'.format(exc_type))
    try:
        exc = _get_executor(exc_type, arg)
    except Exception as err:
        logger.exception('Could not setup an executor of type "{}" '
                         'with args "{}": "{}"'
                         .format(exc_type, arg, err))
        raise
    else:
        return exc


