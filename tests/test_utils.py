""" Tests for `yatsm.utils`
"""
import numpy as np
import pytest

from yatsm import utils


@pytest.mark.parametrize('nrow,njob', [(793, 13), (700, 1), (700, 700)])
def test_distribute_jobs_interlaced(nrow, njob):
    assigned = []
    for i in range(njob):
        assigned.extend(utils.distribute_jobs(i, njob, nrow, interlaced=True))

    assigned = np.sort(np.asarray(assigned))
    all_rows = np.arange(0, nrow)
    np.testing.assert_equal(assigned, all_rows)


@pytest.mark.parametrize('nrow,njob', [(793, 13), (700, 1), (700, 700)])
def test_distribute_jobs_sequential(nrow, njob):
    assigned = []
    for i in range(njob):
        assigned.extend(utils.distribute_jobs(i, njob, nrow, interlaced=False))

    assigned = np.sort(np.asarray(assigned))
    all_rows = np.arange(0, nrow)
    np.testing.assert_equal(assigned, all_rows)


@pytest.mark.parametrize('nrow,njob', [(700, 1)])
def test_distribute_jobs_sequential_onejob(nrow, njob):
    with pytest.raises(ValueError):
        utils.distribute_jobs(nrow, nrow, njob, interlaced=False)


# mkdir_p
def test_mkdir_p_success(tmpdir):
    utils.mkdir_p(tmpdir.join('test').strpath)


def test_mkdir_p_succcess_exists(tmpdir):
    utils.mkdir_p(tmpdir.join('test').strpath)
    utils.mkdir_p(tmpdir.join('test').strpath)


def test_mkdir_p_failure_permission(tmpdir):
    with pytest.raises(OSError):
        utils.mkdir_p('/asdf')


# np_promote_all_types
@pytest.mark.parametrize(('dtypes', 'ans'), [
    ((np.uint8, np.int16), np.int16),
    ((np.uint8, np.uint16, np.int16), np.int32),
    ((np.uint8, np.uint16, np.int16, np.float), np.float),
    ((np.uint8, np.float16, np.float32, np.float64), np.float64),
])
def test_np_promote_all_types(dtypes, ans):
    test_ans = utils.np_promote_all_types(*dtypes)
    assert test_ans == ans
