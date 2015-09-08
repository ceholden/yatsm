#!/usr/bin/env python
""" Tests for `yatsm.cache`
"""
import logging
import os
import unittest

import numpy as np

from utils_testing import create_dir, remove_dir, TestStackDataset
from yatsm import cache, reader
from yatsm.log_yatsm import logger

logger.setLevel(logging.DEBUG)


class TestCache(TestStackDataset):

    @classmethod
    def setUpClass(cls):
        """ Setup test data """
        super(TestCache, cls).setUpClass()
        # Setup a config dict with relevant info
        cls.config = {}
        cls.config['cache_line_dir'] = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'cache')

        # Setup attributes for example cache dataset
        cls.n_images = 447
        cls.n_bands = 8
        cls.n_row = 0

        # Test output file
        cls.test_file = os.path.join(cls.config['cache_line_dir'],
                                     'yatsm_r0_n447_b8.npy.npz')
        cls.test_data = np.load(cls.test_file)
        cls.test_stack = 'data/subset'

    def tearDown(self):
        pass

    def test_get_line_cache_name(self):
        name = cache.get_line_cache_name(
            self.config, self.n_images, self.n_row, self.n_bands)
        self.assertEqual(name, self.test_file)

    def test_get_line_cache_pattern_glob(self):
        import glob
        pattern = cache.get_line_cache_pattern(
            self.n_row, self.n_bands, regex=False)

        found = glob.glob('{d}/{p}'.format(
            d=self.config['cache_line_dir'], p=pattern))[0]

        self.assertEqual(found, self.test_file)

    def test_get_line_cache_pattern_regex(self):
        import re
        pattern = cache.get_line_cache_pattern(
            self.n_row, self.n_bands, regex=True)

        found = [f for f in os.listdir(self.config['cache_line_dir'])
                 if re.match(pattern, f)]
        found = os.path.join(self.config['cache_line_dir'], found[0])

        self.assertEqual(found, self.test_file)

    def test_cache_permissions_readF_writeF(self):
        # False / False
        test_dir = 'test/test_f_f'
        create_dir(test_dir, read=False, write=False)
        self.assertEqual((False, False),
                         cache.test_cache({'cache_line_dir': test_dir}))
        remove_dir(test_dir)

    def test_cache_permissions_readT_writeF(self):
        # True / False
        test_dir = 'test/test_t_f'
        create_dir(test_dir, read=True, write=False)
        self.assertEqual((True, False),
                         cache.test_cache({'cache_line_dir': test_dir}))
        remove_dir(test_dir)

    def test_cache_permissions_readF_writeT(self):
        # False / True
        test_dir = 'test/test_f_t'
        create_dir(test_dir, read=False, write=True)
        self.assertEqual((False, True),
                         cache.test_cache({'cache_line_dir': test_dir}))
        remove_dir(test_dir)

    def test_cache_permissions_readT_writeT(self):
        # False / False
        test_dir = 'test/test_t_t'
        create_dir(test_dir, read=True, write=True)
        self.assertEqual((True, True),
                         cache.test_cache({'cache_line_dir': test_dir}))
        remove_dir(test_dir)

    def test_cache_permissions_create(self):
        test_dir = 'test/'
        self.assertEqual((True, True),
                         cache.test_cache({'cache_line_dir': test_dir}))
        remove_dir(test_dir)

    def test_read_cache_file(self):
        # Expect None from non-existent file
        self.assertIsNone(cache.read_cache_file('asdf'))

        # Expect correct data without image ID check
        np.testing.assert_equal(self.test_data['Y'],
                                cache.read_cache_file(self.test_file))

    def test_read_cache_file_imageIDs(self):
        # Expect None because image IDs won't match
        image_IDs = self.test_data['image_IDs'][:-1]

        self.assertIsNone(cache.read_cache_file(self.test_file, image_IDs))

        # Expect correct data with image ID check
        np.testing.assert_equal(
            self.test_data['Y'],
            cache.read_cache_file(self.test_file,
                                  self.test_data['image_IDs']))

    def test_write_cache_file(self):
        cache.write_cache_file('test_write_1.npz',
                               self.test_data['Y'],
                               self.test_data['image_IDs'])
        test = np.load('test_write_1.npz')
        np.testing.assert_equal(test['Y'], self.test_data['Y'])
        np.testing.assert_equal(test['image_IDs'], self.test_data['image_IDs'])
        os.remove('test_write_1.npz')

    def test_update_cache_file_delete_obs(self):
        choice = np.random.choice(self.test_data['image_IDs'].size,
                                  size=100, replace=False)
        new_Y = self.test_data['Y'][:, choice, :]
        new_image_IDs = self.test_data['image_IDs'][choice]

        # For now, just use image_IDs as `images` since we won't be updating
        # from images
        cache.update_cache_file(new_image_IDs, new_image_IDs,
                                self.test_file,
                                'test_write_2.npz',
                                0, reader.read_row_GDAL)

        new_cache = np.load('test_write_2.npz')

        np.testing.assert_equal(new_Y, new_cache['Y'])
        np.testing.assert_equal(new_image_IDs, new_cache['image_IDs'])

        os.remove('test_write_2.npz')

    def test_update_cache_file_add_obs(self):
        """ Grab a subset of test data and see if we get more data back """
        # Presort and subset for comparison
        sort_idx = np.argsort(self.test_data['image_IDs'])
        test_Y = self.test_data['Y'][:, sort_idx, :]
        test_IDs = self.test_data['image_IDs'][sort_idx]

        size_1 = 100
        size_2 = 200

        sort_idx = np.argsort(self.stack_image_IDs)[:size_2]
        stack_images = self.stack_images[sort_idx]
        stack_IDs = self.stack_image_IDs[sort_idx]

        # Create reduced dataset to add to
        np.savez_compressed('test_write_3.npz',
                            Y=test_Y[:, :size_1, :],
                            image_IDs=test_IDs[:size_1])

        # Write update and read back
        cache.update_cache_file(stack_images, stack_IDs,
                                'test_write_3.npz', 'test_write_new_3.npz',
                                0, reader.read_row_GDAL)
        updated = np.load('test_write_new_3.npz')

        # Test and clean update
        np.testing.assert_equal(test_Y[:, :size_2, :], updated['Y'])
        np.testing.assert_equal(test_IDs[:size_2], updated['image_IDs'])

        os.remove('test_write_3.npz')
        os.remove('test_write_new_3.npz')


if __name__ == '__main__':
    unittest.main()
