""" Tests for ``yatsm.executor``
"""
import math

from concurrent.futures import Future
HAS_DISTRIBUTED = True
try:
    import distributed  # noqa
except ImportError:
    HAS_DISTRIBUTED = False
import pytest  # noqa

from yatsm import executor  # noqa

req_distributed = pytest.mark.skipif(not HAS_DISTRIBUTED,
                                     reason="Requires dask.distributed")

maths = {
    'q': (math.sin, 2.0 * math.pi / 180.0 * 45.0, ),
    'a': 1.0
}


def test_get_executor_sync():
    exc = executor.get_executor('sync', None)
    assert isinstance(exc, executor.SyncExecutor)
    future = exc.submit(*maths['q'])
    assert isinstance(future, Future)
    assert exc.result(future) == maths['a']


def test_get_executor_thread():
    exc = executor.get_executor('thread', 1)
    assert isinstance(exc, executor.ConcurrentExecutor)


def test_get_executor_process():
    exc = executor.get_executor('process', 1)
    assert isinstance(exc, executor.ConcurrentExecutor)


@req_distributed
def test_get_executor_distributed(cluster):
    exc = executor.get_executor('distributed', cluster.scheduler_address)
    assert isinstance(exc, executor.DistributedExecutor)


@req_distributed
def test_get_executor_unknown(cluster):
    with pytest.raises(KeyError, message="Unknown executor") as err:
        executor.get_executor('asdf', None)
    assert 'Unknown executor' in str(err.value)


@pytest.fixture(scope='session')
def cluster(request):
    try:
        from distributed import LocalCluster
    except ImportError:
        yield None
    else:
        cluster = LocalCluster()
        yield cluster
        cluster.close()
