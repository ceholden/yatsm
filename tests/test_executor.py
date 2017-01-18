""" Tests for ``yatsm.executor``
"""
import inspect

import click
import dask
HAS_DISTRIBUTED = True
try:
    import distributed
except ImportError:
    HAS_DISTRIBUTED = False
import pytest

from yatsm.executor import get_executor

req_distributed = pytest.mark.skipif(not HAS_DISTRIBUTED,
                                     reason="Requires dask.distributed")


def test_get_executor_sync():
    result = get_executor('sync', None)
    assert result is dask.async.get_sync


@req_distributed
def test_get_executor_distributed(cluster):
    result = get_executor('distributed', cluster.scheduler_address)
    assert get_owner_class(result) is distributed.Client
    assert result.__name__ == 'get'


@req_distributed
def test_get_executor_unknown(cluster):
    with pytest.raises(KeyError, message="Gave unknown executor type") as err:
        result = get_executor('asdf', None)
    assert 'Unknown executor' in str(err.value)


@pytest.fixture(scope='session')
def cluster(request):
    try:
        from distributed import LocalCluster
    except ImportError:
        return None
    else:
        return LocalCluster()


def get_owner_class(meth):
    mod = inspect.getmodule(meth)
    name = (getattr(meth, '__qualname__',
                    getattr(meth, 'im_class', '').__name__)
            .rsplit('.', 1)[0])
    return getattr(mod, name)
