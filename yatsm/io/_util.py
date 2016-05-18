""" Collection of  functions that ease common filesystem operations
"""
import errno
import os


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
