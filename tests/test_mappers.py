#!/usr/bin/env python
""" Tests for YATSM scripts
"""
import os
import shutil
import unittest

from utils_testing import TestMaps


class Test_YATSMMap(TestMaps):

    def setUp(self):
        """ Setup test data filenames and load known truth dataset """
        self.script = 'yatsm_map.py'

        # Test data
        self.root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        self.result_dir = os.path.join(self.root, 'YATSM')
        self.robust_result_dir = os.path.join(self.root, 'YATSM_ROBUST')
        self.data_cache = os.path.join(self.root, 'cache')
        self.example_img = os.path.join(self.root, 'example_img')
        self.outdir = os.path.join(self.root, 'outdir')

        # Answers
        self.answers = os.path.join(self.root, 'answers')

        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def tearDown(self):
        """ Deletes answer directory """
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

# Test coefficients
    def test_coef(self):
        """ Test creating coefficient map """
        output = os.path.join(self.outdir, 'coef_all.gtif')

        args = '--root {r} coef 2000-06-01 {o}'.format(
            r=self.root,
            o=output).split(' ')

        msg, retcode = self._run(self.script, args)
        self.assertEqual(retcode, 0)

        # Test output
        self._compare_maps(os.path.join(self.outdir, output),
                           os.path.join(self.answers, output))

    def test_coef_robust(self):
        """ Test creating robust coefficient map """
        output = 'coef_all_robust.gtif'

        # Test robust coefficients
        args = '--root {r} --result {rr} --robust coef 2000-06-01 {o}'.format(
            r=self.root,
            rr=self.robust_result_dir,
            o=os.path.join(self.outdir, output)).split(' ')

        msg, retcode = self._run(self.script, args)
        self.assertEqual(retcode, 0)

        # Test robust coefficients, expecting error
        args = '--root {r} --robust coef 2000-06-01 {o}'.format(
            r=self.root,
            o=os.path.join(self.outdir, output)).split(' ')
        msg, retcode = self._run(self.script, args)

        self.assertEqual(retcode, 1)

        # Test output
        self._compare_maps(os.path.join(self.outdir, output),
                           os.path.join(self.answers, output))

    def test_coef_bands(self):
        """ Test if correct bands are output """
        # Test bands outputs
        # Test coefficient outputs
        pass

    def test_coef_before_after(self):
        """ Test use of --before and --after flags """
        pass

# Test prediction
# Test classification
# Test optional arguments
    def test_ndv(self):
        """ Test output file NoDataValue """
        pass

    def test_output_format(self):
        """ Test output GDAL file format """
        pass

    def test_date_format(self):
        """ Test input date format type """
        pass


class Test_YATSMChangeMap(TestMaps):

    def setUp(self):
        """ Setup test data filenames and load known truth dataset """
        self.script = 'yatsm_changemap.py'

        # Test data
        self.root = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        self.result_dir = os.path.join(self.root, 'YATSM')
        self.robust_result_dir = os.path.join(self.root, 'YATSM_ROBUST')
        self.data_cache = os.path.join(self.root, 'cache')
        self.example_img = os.path.join(self.root, 'example_img')
        self.outdir = os.path.join(self.root, 'outdir')

        # Answers
        self.answers = os.path.join(self.root, 'answers')

        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

    def tearDown(self):
        """ Deletes answer directory """
        if os.path.isdir(self.outdir):
            shutil.rmtree(self.outdir)

    def test_numchange(self):
        pass


if __name__ == '__main__':
    unittest.main()
