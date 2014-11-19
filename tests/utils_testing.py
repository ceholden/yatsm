""" Helper class & functions for testing
"""
import subprocess
import unittest

import numpy as np
from osgeo import gdal


class TestMaps(unittest.TestCase):
    """ Helper class for testing map attributes and values
    """

    def _compare_maps(self, test, truth,
                      num_test_almost_eq=False):
        """ Compare map attributes and values

        Args:
          test (str): filename of data to test
          truth (str): filename containing 'truth'
          num_test_almost_eq (bool, optional): test

        """
        test_ds = gdal.Open(test, gdal.GA_ReadOnly)
        truth_ds = gdal.Open(truth, gdal.GA_ReadOnly)

        # Test dimensions
        self.assertEqual(test_ds.RasterCount, truth_ds.RasterCount,
                         'Number of bands differ')
        self.assertEqual(test_ds.RasterXSize, truth_ds.RasterXSize,
                         'Number of columns differ')
        self.assertEqual(test_ds.RasterYSize, truth_ds.RasterYSize,
                         'Number of rows differ')

        # Test projection / geotransform
        self.assertEqual(test_ds.GetGeoTransform(), truth_ds.GetGeoTransform(),
                         'GeoTransform differs')
        self.assertEqual(test_ds.GetProjection(), truth_ds.GetProjection(),
                         'Projection differs')

        # Test values
        for b in range(1, test_ds.RasterCount + 1):
            test_arr = test_ds.GetRasterBand(b).ReadAsArray()
            truth_arr = truth_ds.GetRasterBand(b).ReadAsArray()

            if num_test_almost_eq:
                np.testing.assert_array_almost_equal(
                    test_arr, truth_arr,
                    err_msg='Band {0} is not almost equal'.format(b)
                )
            else:
                np.testing.assert_array_equal(
                    test_arr, truth_arr,
                    err_msg='Band {0} is not equal'.format(b)
                )

    def _run(self, script, args):
        """ Use subprocess to run script with arguments

        Args:
          script (str): script filename to run
          args (list): program arguments

        Returns:
          tuple: stdout and exit code

        """
        proc = subprocess.Popen([script] + args,
                                stdout=subprocess.PIPE
                                )

        stdout = proc.communicate()[0]
        retcode = proc.returncode

        return stdout, retcode
