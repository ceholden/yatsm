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
