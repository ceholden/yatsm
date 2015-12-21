""" Tests for ``yatsm.io.helpers``
"""
import pytest

from yatsm.io import helpers


def test_mkdir_p_success(tmpdir):
    helpers.mkdir_p(tmpdir.join('test').strpath)


def test_mkdir_p_succcess_exists(tmpdir):
    helpers.mkdir_p(tmpdir.join('test').strpath)
    helpers.mkdir_p(tmpdir.join('test').strpath)


def test_mkdir_p_failure_permission(tmpdir):
    with pytest.raises(OSError):
        helpers.mkdir_p('/asdf')
