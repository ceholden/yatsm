""" Tests for :ref:`yatsm.tslib`
"""
import numpy as np
import pytest

from yatsm.tslib import datetime2int


# DATETIME
@pytest.mark.parametrize(('in_', 'out_', 'out_format', 'in_format'), (
    (np.array([734910]), np.array([20130211]), '%Y%m%d', None),
    (np.array([734910]), np.array([734910]), 'ordinal', None),
))
def test_datetime2int(in_, out_, out_format, in_format):
    ans = datetime2int(in_, out_format=out_format, in_format=in_format)
    assert (ans == out_).all()
