""" Utility functions
"""
from __future__ import division

import fnmatch
from functools import wraps
import logging
import os
import re

import six

try:
    from scandir import walk
except:
    from os import walk

logger = logging.getLogger(__name__)


def to_number(string):
    """ Convert string to most appropriate number
    """
    try:
        n = int(string)
    except ValueError:
        n = float(string)
    return n


# JOB SPECIFIC FUNCTIONS
def distribute_jobs(job_number, total_jobs, n, interlaced=True):
    """ Assign `job_number` out of `total_jobs` a subset of `n` tasks

    Args:
        job_number (int): 0-indexed processor to distribute jobs to
        total_jobs (int): total number of processors running jobs
        n (int): number of tasks (e.g., lines in image, regions in segment)
        interlaced (bool, optional): interlace job assignment (default: True)

    Returns:
        np.ndarray: np.ndarray of task IDs to be processed

    Raises:
        ValueError: raise error if `job_number` and `total_jobs` specified
            result in no jobs being assinged (happens if `job_number` and
            `total_jobs` are both 1)

    """
    import numpy as np
    if interlaced:
        assigned = 0
        tasks = []

        while job_number + total_jobs * assigned < n:
            tasks.append(job_number + total_jobs * assigned)
            assigned += 1
        tasks = np.asarray(tasks)
    else:
        size = int(n / total_jobs)
        i_start = size * job_number
        i_end = size * (job_number + 1)

        tasks = np.arange(i_start, min(i_end, n))

    if tasks.size == 0:
        raise ValueError(
            'No jobs assigned for job_number/total_jobs: {j}/{t}'.format(
                j=job_number,
                t=total_jobs))

    return tasks


def find(location, pattern, regex=False):
    """ Return a sorted list of files matching pattern

    Args:
        location (str): Directory location to search
        pattern (str): Search pattern for files
        regex (bool): True if ``pattern`` is a regular expression

    Returns:
        list: List of file paths for files found

    """
    if not regex:
        pattern = fnmatch.translate(pattern)
    regex = re.compile(pattern)

    files = []
    for root, dirnames, filenames in walk(location):
        for filename in filenames:
            if regex.search(filename):
                files.append(os.path.join(root, filename))

    return sorted(files)


# MISC UTILITIES
def copy_dict_filter_key(d, regex):
    """ Copy a dict recursively, but only if key doesn't match regex pattern
    """
    out = {}
    for k, v in six.iteritems(d):
        if not re.match(regex, k):
            if isinstance(v, dict):
                out[k] = copy_dict_filter_key(v, regex)
            else:
                out[k] = v
    return out


def cached_property(prop):
    """ Cache a class property (e.g., that requires a lookup)
    """
    prop_name = '_{}'.format(prop.__name__)

    @wraps(prop)
    def wrapper(self):
        if not hasattr(self, prop_name):
            setattr(self, prop_name, prop(self))
        return getattr(self, prop_name)

    return property(wrapper)
