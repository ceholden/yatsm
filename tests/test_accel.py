import os

import numpy as np
import pytest

from yatsm import accel

has_numba = True
try:
    import numba as nb
except ImportError:
    has_numba = False

# Alter expected JIT'd function class based on environment variable
if os.environ.get('NUMBA_DISABLE_JIT') is not None:
    nb.config.DISABLE_JIT = 1
    jitd_class = type(nb.jit(lambda x: x ** 2))
    jit_enabled = False
else:
    nb.config.DISABLE_JIT = 0
    jitd_class = type(nb.jit(lambda x: x ** 2))
    jit_enabled = True


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
    assert isinstance(fn, jitd_class)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_2():
    """ Use decorator with parentheses"""
    accel.has_numba = True

    @accel.try_jit()
    def fn():
        return np.ones(100) * 5
    assert isinstance(fn, jitd_class)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_3():
    """ Use decorator with kwargs"""
    accel.has_numba = True

    @accel.try_jit(nopython=True)
    def fn():
        return np.ones(100) * 5
    fn()
    assert isinstance(fn, jitd_class)
    if jit_enabled:
        assert len(fn.nopython_signatures) > 0


@pytest.mark.skipif("not has_numba")
def test_accel_nb_4(fn):
    """ JIT with function """
    accel.has_numba = True
    assert isinstance(accel.try_jit(fn), jitd_class)


@pytest.mark.skipif("not has_numba")
def test_accel_nb_5(fn):
    """ JIT with function, with kwargs"""
    accel.has_numba = True
    fn = accel.try_jit(fn, nopython=True)
    fn()
    assert isinstance(fn, jitd_class)
    if jit_enabled:
        assert len(fn.nopython_signatures) > 0


def test_accel_no_nb(fn):
    """ Test function isn't JIT-ed if no numba """
    accel.has_numba = False
    assert accel.try_jit(fn) is fn
