import fnmatch
from functools import partial
import os
import shutil
import tarfile
from tempfile import mkdtemp

try:
    from os import walk
except ImportError:
    from scandir import walk

import numpy as np
import pytest

here = os.path.dirname(__file__)
example_cachedir = os.path.join(here, 'data', 'cache')
example_cachefile = os.path.join(example_cachedir, 'yatsm_r0_n447_b8.npy.npz')


# EXAMPLE DATASETS
@pytest.fixture(scope='session')
def example_timeseries(request):
    """ Extract example timeseries returning paths & image IDs
    """
    path = mkdtemp('_yatsm')
    tgz = os.path.join(here, 'data', 'p035r032_subset.tar.gz')
    with tarfile.open(tgz) as tgz:
        tgz.extractall(path)
    request.addfinalizer(partial(shutil.rmtree, path))

    subset_path = os.path.join(path, 'subset')
    stack_images, stack_image_IDs = [], []
    for root, dnames, fnames in walk(subset_path):
        for fname in fnmatch.filter(fnames, 'L*stack'):
            stack_images.append(os.path.join(root, fname))
            stack_image_IDs.append(os.path.basename(root))

    return subset_path, np.asarray(stack_images), np.asarray(stack_image_IDs)


@pytest.fixture(scope='session')
def example_cache(request):
    return np.load(example_cachefile)


# EXAMPLE CACHE DATA
@pytest.fixture(scope='function')
def cachedir(request):
    return example_cachedir


@pytest.fixture(scope='function')
def cachefile(request):
    return example_cachefile


# MISC
@pytest.fixture(scope='function')
def mkdir_permissions(request):
    """ Fixture for creating dir with specific read/write permissions """
    def make_mkdir(read=False, write=False):
        if read and write:
            mode = 0755
        elif read and not write:
            mode = 0555
        elif not read and write:
            mode = 0333
        elif not read and not write:
            mode = 0000

        path = mkdtemp()
        os.chmod(path, mode)

        def fin():
            os.chmod(path, 0755)
            os.removedirs(path)
        request.addfinalizer(fin)

        return path

    return make_mkdir
