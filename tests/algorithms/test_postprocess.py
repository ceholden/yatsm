""" Test postprocessing algorithms
"""
import numpy as np

from yatsm.algorithms.postprocess import commission_test


def test_commission_nochange(sim_nochange):
    """ In no change situation, we should get back exactly what we gave in
    """
    record = commission_test(sim_nochange, 0.10)
    assert len(record) == 1
    np.testing.assert_array_equal(record, sim_nochange.record)


def test_commission_no_real_change(sim_no_real_change):
    """ Test commission test's ability to resolve spurious change
    """
    record = commission_test(sim_no_real_change, 0.01)
    assert len(record) == 1
    assert record[0]['break'] == 0


def test_commission_real_change(sim_real_change):
    """ Test commission test's ability to avoid merging real changes

    This test is run with a relatively large p value (very likely to reject H0
    and retain changes)
    """
    record = commission_test(sim_real_change, 0.10)
    assert len(record) == len(sim_real_change.record)
