""" Tests for ``yatsm.io._util``
"""
import pytest

from yatsm.io import _util


def test_mkdir_p_success(tmpdir):
    _util.mkdir_p(tmpdir.join('test').strpath)


def test_mkdir_p_succcess_exists(tmpdir):
    _util.mkdir_p(tmpdir.join('test').strpath)
    _util.mkdir_p(tmpdir.join('test').strpath)


def test_mkdir_p_failure_permission(tmpdir):
    with pytest.raises(OSError):
        _util.mkdir_p('/asdf')
