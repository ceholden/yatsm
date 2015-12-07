import numpy as np
import pytest

from yatsm import accel

has_numba = True
try:
    import numba as nb
except ImportError:
    has_numba = False


@pytest.fixture
def fn():
    def func():
        return np.ones(100) * 5
    return func


@pytest.mark.skipif("not has_numba")
def test_accel_nb_1():
    """ Use decorator """
    accel.has_numba = True

    @accel.try_jit
    def fn():
        return np.ones(100) * 5
    assert isinstance(fn, nb.targets.registry.CPUOverloaded)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_2():
    """ Use decorator with parentheses"""
    accel.has_numba = True

    @accel.try_jit()
    def fn():
        return np.ones(100) * 5
    assert isinstance(fn, nb.targets.registry.CPUOverloaded)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_3():
    """ Use decorator with kwargs"""
    accel.has_numba = True

    @accel.try_jit(nopython=True)
    def fn():
        return np.ones(100) * 5
    assert isinstance(fn, nb.targets.registry.CPUOverloaded)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_4(fn):
    """ JIT with function """
    accel.has_numba = True
    assert isinstance(accel.try_jit(fn), nb.targets.registry.CPUOverloaded)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_5(fn):
    """ JIT with function, with kwargs"""
    accel.has_numba = True
    assert isinstance(accel.try_jit(fn, nopython=True),
                      nb.targets.registry.CPUOverloaded)


def test_accel_no_nb(fn):
    """ Test function isn't JIT-ed if no numba """
    accel.has_numba = False
    assert accel.try_jit(fn) is fn
