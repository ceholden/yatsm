""" Tests for yatsm.cache
"""
import os
import tempfile

import numpy as np
import pytest
from yatsm import cache, reader

cache_params = pytest.mark.parametrize('n_images,n_row,n_bands', [(447, 0, 8)])


@cache_params
def test_get_line_cache_name(cachedir, cachefile, n_images, n_row, n_bands):
    cfg = dict(cache_line_dir=cachedir)
    assert cachefile == cache.get_line_cache_name(cfg,
                                                  n_images, n_row, n_bands)


@cache_params
def test_get_line_cache_pattern_glob(cachedir, cachefile,
                                     n_images, n_row, n_bands):
    import glob
    pattern = cache.get_line_cache_pattern(n_row, n_bands, regex=False)
    found = glob.glob('%s/%s' % (cachedir, pattern))[0]

    assert found == cachefile


@cache_params
def test_get_line_cache_pattern_regex(cachedir, cachefile,
                                      n_images, n_row, n_bands):
    import re
    pattern = cache.get_line_cache_pattern(n_row, n_bands, regex=True)

    found = [f for f in os.listdir(cachedir) if re.match(pattern, f)]
    found = os.path.join(cachedir, found[0])

    assert found == cachefile


def test_test_cache(mkdir_permissions):
    # Test when cache dir exists already
    path = mkdir_permissions(read=False, write=False)
    assert (False, False) == cache.test_cache(dict(cache_line_dir=path))

    path = mkdir_permissions(read=False, write=True)
    assert (False, True) == cache.test_cache(dict(cache_line_dir=path))

    path = mkdir_permissions(read=True, write=False)
    assert (True, False) == cache.test_cache(dict(cache_line_dir=path))

    path = mkdir_permissions(read=True, write=True)
    assert (True, True) == cache.test_cache(dict(cache_line_dir=path))

    # Test when cache dir doesn't exist
    tmp = os.path.join(tempfile.tempdir,
                       next(tempfile._get_candidate_names()) + '_yatsm')
    read_write = cache.test_cache(dict(cache_line_dir=tmp))
    os.removedirs(tmp)

    assert (True, True) == read_write


def test_read_cache_file(cachefile, example_cache):
    assert None is cache.read_cache_file('asdf')

    np.testing.assert_equal(example_cache['Y'],
                            cache.read_cache_file(cachefile))


def test_read_cache_file_imageIDs(cachefile, example_cache):
    image_IDs = example_cache['image_IDs']
    # Expect None since image IDs won't match
    assert None is cache.read_cache_file(cachefile,
                                         image_IDs[::-1])

    np.testing.assert_equal(example_cache['Y'],
                            cache.read_cache_file(cachefile, image_IDs))


def test_write_cache_file(cachefile, example_cache):
    cache.write_cache_file('test.npz',
                           example_cache['Y'], example_cache['image_IDs'])
    test = np.load('test.npz')
    Y, image_IDs = test['Y'], test['image_IDs']
    os.remove('test.npz')

    np.testing.assert_equal(Y, example_cache['Y'])
    np.testing.assert_equal(image_IDs, example_cache['image_IDs'])


def test_update_cache_file_delete_obs(cachefile, example_cache):
    choice = np.random.choice(example_cache['image_IDs'].size,
                              size=100, replace=False)
    new_Y = example_cache['Y'][:, choice, :]
    new_image_IDs = example_cache['image_IDs'][choice]

    # For now, just use image_IDs as `images` since we won't be updating
    # from images
    cache.update_cache_file(new_image_IDs, new_image_IDs,
                            cachefile,
                            'test.npz',
                            0, reader.read_row_GDAL)
    test = np.load('test.npz')
    Y, image_IDs = test['Y'], test['image_IDs']
    os.remove('test.npz')

    np.testing.assert_equal(new_Y, Y)
    np.testing.assert_equal(new_image_IDs, image_IDs)


def test_update_cache_file_add_obs(cachefile, example_cache,
                                   example_timeseries):
    """ Grab a subset of test data and see if we get more data back """
    path, stack_images, stack_image_IDs = example_timeseries
    # Presort and subset for comparison
    sort_idx = np.argsort(example_cache['image_IDs'])
    test_Y = example_cache['Y'][:, sort_idx, :]
    test_IDs = example_cache['image_IDs'][sort_idx]

    size_1 = 100
    size_2 = 200

    sort_idx = np.argsort(stack_image_IDs)[:size_2]
    stack_images = stack_images[sort_idx]
    stack_IDs = stack_image_IDs[sort_idx]

    # Create reduced dataset to add to
    np.savez_compressed('test.npz',
                        Y=test_Y[:, :size_1, :],
                        image_IDs=test_IDs[:size_1])

    # Write update and read back
    cache.update_cache_file(stack_images, stack_IDs,
                            'test.npz', 'test_new.npz',
                            0, reader.read_row_GDAL)
    updated = np.load('test_new.npz')

    # Test and clean update
    np.testing.assert_equal(test_Y[:, :size_2, :], updated['Y'])
    np.testing.assert_equal(test_IDs[:size_2], updated['image_IDs'])

    os.remove('test.npz')
    os.remove('test_new.npz')
