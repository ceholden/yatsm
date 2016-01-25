""" Collection of helper functions that ease common filesystem operations
"""
import datetime as dt
import errno
import fnmatch
import os
try:
    from scandir import walk
except ImportError:  # Py3.5+ and incorporated or Py2/<Py3.5
    from os import walk


def mkdir_p(d):
    """ Make a directory, ignoring error if it exists (i.e., ``mkdir -p``)

    Args:
        d (str): directory path to create

    Raises:
        OSError: Raise OSError if cannot create directory for reasons other
            than it existing already (errno 13 "EEXIST")
    """
    try:
        os.makedirs(d)
    except OSError as err:
        # File exists
        if err.errno == errno.EEXIST:
            pass
        else:
            raise err


def find_stack_images(location, folder_pattern='L*', image_pattern='L*stack',
                      date_index_start=9, date_index_end=16,
                      date_format='%Y%j',
                      ignore=['YATSM']):
    """ Find and identify dates and filenames of Landsat image stacks

    Args:
        location (str): Stacked image dataset location
        folder_pattern (str, optional): Filename pattern for stack image
            folders located within `location` (default: 'L*')
        image_pattern (str, optional): Filename pattern for stacked images
            located within each folder (default: 'L*stack')
        date_index_start (int, optional): Starting index of image date string
            within folder name (default: 9)
        date_index_end (int, optional): Ending index of image date string
            within folder name (default: 16)
        date_format (str, optional): String format of date within folder names
            (default: '%Y%j')
        ignore (list, optional): List of folder names within `location` to
            ignore from search (default: ['YATSM'])

    Returns:
        tuple: Tuple of lists containing the dates and filenames of all stacked
            images located
    """
    if isinstance(ignore, str):
        ignore = [ignore]

    folder_names = []
    image_filenames = []
    dates = []

    # Populate - only checking one directory down
    location = location.rstrip(os.path.sep)
    num_sep = location.count(os.path.sep)

    for root, dnames, fnames in walk(location, followlinks=True):
        # Remove results folder
        dnames[:] = [d for d in dnames for i in ignore if i not in d]

        # Force only 1 level
        num_sep_this = root.count(os.path.sep)
        if num_sep + 1 <= num_sep_this:
            del dnames[:]

        # Directory names as image IDs
        for dname in fnmatch.filter(dnames, folder_pattern):
            folder_names.append(dname)

        # Add file name and paths
        for fname in fnmatch.filter(fnames, image_pattern):
            image_filenames.append(os.path.join(root, fname))

    # Check to see if we found anything
    if not folder_names or not image_filenames:
        raise Exception('Zero stack images found with image '
                        'and folder patterns: {0}, {1}'.format(
                            folder_pattern, image_pattern))

    if len(folder_names) != len(image_filenames):
        raise Exception(
            'Inconsistent number of stacks folders and stack images located')

    # Extract dates
    for folder in folder_names:
        dates.append(dt.datetime.strptime(
            folder[date_index_start:date_index_end], date_format))

    # Sort images by date
    dates, image_filenames = (
        list(t) for t in
        zip(*sorted(zip(dates, image_filenames)))
    )

    return (dates, image_filenames)
