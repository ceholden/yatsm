""" Tests for ``yatsm.executor``
"""
import inspect
import math
import operator

import click
from concurrent.futures import Future
import dask
HAS_DISTRIBUTED = True
try:
    import distributed
except ImportError:
    HAS_DISTRIBUTED = False
import pytest

from yatsm import executor

req_distributed = pytest.mark.skipif(not HAS_DISTRIBUTED,
                                     reason="Requires dask.distributed")

maths = {
    'q': (math.sin, 2.0 * math.pi / 180.0 * 45.0, ),
    'a': 1.0
}


def test_get_executor_sync():
    result = executor.get_executor('sync', None)
    assert isinstance(result, executor.SyncExecutor)
    future = result.submit(*maths['q'])
    assert isinstance(future, Future)
    assert future.result() == maths['a']


def test_get_executor_thread():
    result = executor.get_executor('thread', 1)
    assert isinstance(result, executor.ConcurrentExecutor)


def test_get_executor_process():
    result = executor.get_executor('process', 1)
    assert isinstance(result, executor.ConcurrentExecutor)


@req_distributed
def test_get_executor_distributed(cluster):
    result = executor.get_executor('distributed', cluster.scheduler_address)
    assert isinstance(result, executor.DistributedExecutor)


@req_distributed
def test_get_executor_unknown(cluster):
    with pytest.raises(KeyError, message="Unknown executor") as err:
        result = executor.get_executor('asdf', None)
    assert 'Unknown executor' in str(err.value)


@pytest.fixture(scope='session')
def cluster(request):
    try:
        from distributed import LocalCluster
    except ImportError:
        return None
    else:
        return LocalCluster()
