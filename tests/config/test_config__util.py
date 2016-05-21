""" Tests for yatsm.io._util.py
"""
from contextlib import contextmanager
import os

import pytest

import yatsm.config._util as util


# TODO: extend_with_default


# ENVIRONMENT VARIABLE PARSING
@contextmanager
def _rollback_env(update):
    backup = os.environ.copy()
    os.environ.update(update)
    yield
    os.environ = backup


@pytest.mark.parametrize(('envvar', 'provided', 'expected'), [
    ({'ROOT': '/root'}, {'path': '$ROOT/test'}, {'path': '/root/test'}),
    ({'A': '1'}, {'MIN': '[$A, $A, 10, $A]'}, {'MIN': '[1, 1, 10, 1]'}),
    ({'A': '5'}, {'1': {'2': '$A'}, '3': '$A'}, {'1': {'2': '5'}, '3': '5'})
])
def test_get_envvars(envvar, provided, expected):
    with _rollback_env(envvar):
        test = util.expand_envvars(provided)
    assert test == expected
