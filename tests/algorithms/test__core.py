""" Tests for yatsm.algorithms._core
"""
import numpy as np
import pytest

from yatsm.algorithms._core import Segment, SEGMENT_ATTRS


@pytest.mark.parametrize('d', [
    dict(a=0, b=0, c=np.random.rand(5, 5))
])
def test_Segment(d):
    s = Segment.from_example(**d)
    assert all([k in s.dtype.names for k in d])
    assert all([seg_attr in s.dtype.names for seg_attr in SEGMENT_ATTRS])
