import fnmatch
import os
import unittest
import subprocess

import numpy as np


def create_dir(d, read=True, write=True):
    """ Creates a directory with given permissions """
    if read and write:
        mode = 0755
    elif read and not write:
        mode = 0555
    elif not read and write:
        mode = 0333
    elif not read and not write:
        mode = 0000

    if os.path.exists(d):
        remove_dir(d)
    os.makedirs(d)
    os.chmod(d, mode)


def remove_dir(d):
    """ Makes sure we can remove directory by changing permissions first """
    os.chmod(d, 0755)
    os.removedirs(d)


# Utility to ensure we only unzip timeseries dataset once
_here = os.path.join(os.path.dirname(os.path.abspath(__file__)))
_tgz = os.path.join(_here, 'data', 'p035r032_subset.tar.gz')
_to_dir = os.path.join(_here, 'data')
_dir = os.path.join(_here, 'data', 'subset')
_pattern = 'L*stack'


class TestStackDataset(unittest.TestCase):
    """ Special class that ensures test stacked dataset only unzips once """

    @classmethod
    def setUpClass(cls):
        subprocess.call(['tar', '-xzf', _tgz, '-C', _to_dir])
        cls.stack_images = []
        cls.stack_image_IDs = []
        for root, dnames, fnames in os.walk(_dir):
            for fname in fnmatch.filter(fnames, _pattern):
                cls.stack_images.append(os.path.join(root, fname))
                cls.stack_image_IDs.append(os.path.basename(root))

        cls.stack_images = np.asarray(cls.stack_images)
        cls.stack_image_IDs = np.asarray(cls.stack_image_IDs)

    @classmethod
    def tearDownClass(cls):
        subprocess.call(['rm', '-rf', _dir])
