import unittest

import numpy as np

from yatsm import utils


class TestUtils(unittest.TestCase):

    def test_calculate_lines_interlaced(self):
        nrow = 7937
        total_jobs = 13

        assigned = []
        for i in xrange(total_jobs):
            assigned.extend(utils.calculate_lines(i, total_jobs, nrow,
                                                  interlaced=True))

        assigned = np.sort(np.asarray(assigned))
        all_rows = np.arange(0, nrow)

        np.testing.assert_equal(assigned, all_rows)

    def test_calculate_lines_sequential(self):
        nrow = 7937
        total_jobs = 13

        assigned = []
        for i in xrange(total_jobs):
            assigned.extend(utils.calculate_lines(i, total_jobs, nrow,
                                                  interlaced=False))

        assigned = np.asarray(assigned)
        all_rows = np.arange(0, nrow)

        np.testing.assert_equal(assigned, all_rows)

if __name__ == '__main__':
    unittest.main()
