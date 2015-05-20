import unittest

import numpy as np

from yatsm import utils


class TestUtils(unittest.TestCase):

    def test_distribute_jobs_interlaced(self):
        nrow = 7937
        total_jobs = 13

        assigned = []
        for i in xrange(total_jobs):
            assigned.extend(utils.distribute_jobs(i, total_jobs, nrow,
                                                  interlaced=True))

        assigned = np.sort(np.asarray(assigned))
        all_rows = np.arange(0, nrow)

        np.testing.assert_equal(assigned, all_rows)

    def test_distribute_jobs_sequential(self):
        nrow = 7937
        total_jobs = 13

        assigned = []
        for i in xrange(total_jobs):
            assigned.extend(utils.distribute_jobs(i, total_jobs, nrow,
                                                  interlaced=False))

        assigned = np.asarray(assigned)
        all_rows = np.arange(0, nrow)

        np.testing.assert_equal(assigned, all_rows)

    def test_calculate_lines_sequential_onejob(self):
        with self.assertRaises(ValueError):
            utils.distribute_jobs(1, 1, 7937, interlaced=False)


if __name__ == '__main__':
    unittest.main()
