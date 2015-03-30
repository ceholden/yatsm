#!/usr/bin/env python
""" Yet Another Timeseris Model (YATSM) - classify with trained algorithm

Usage:
    classify_yatsm.py [options] <config_file> <trained_algorithm>
        <job_number> <total_jobs>

Options:
    --resume                    Resume classification (don't overwrite)
    -v --verbose                Show verbose debugging messages
    --version                   Print program version and exit
    -h --help                   Show help and exit

"""
from __future__ import division, print_function

import logging
import os
import sys
import time

from docopt import docopt

import numpy as np
import numpy.lib.recfunctions as nprfn

from sklearn.externals import joblib

# Handle running as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.config_parser import parse_config_file
from yatsm.utils import calculate_lines, get_output_name, csvfile_to_dataset
from yatsm.reader import get_image_attribute

FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')


# Helper function
def try_resume(filename):
    """ Return True/False if dataset has already been classified

    Args:
      filename (str): filename of the result to be checked

    Returns:
      bool: If the `npz` file exists and contains a file 'class', this test will
    return True, else False.

    """
    try:
        z = np.load(filename)
    except:
        return False

    if not z['record'].dtype or 'class' not in z['record'].dtype.names:
        return False

    return True


# Main classification function
def classify_line(filename, classifier):
    """ Use `classifier` to classify data stored in `filename`

    Args:
      filename (str): filename of stored results
      classifier (sklearn classifier): pre-trained classifier

    """
    z = np.load(filename)
    rec = z['record']

    if rec.shape[0] == 0:
        logger.debug('No records in {f}. Continuing'.format(f=filename))
        return

    # Rescale intercept term
    coef = rec['coef']
    coef[:, 0, :] = (coef[:, 0, :] + coef[:, 1, :] *
                     ((rec['start'] + rec['end']) / 2.0)[:, np.newaxis])

    # Include RMSE for full X matrix
    newdim = (coef.shape[0], coef.shape[1] * coef.shape[2])
    X = np.hstack((coef.reshape(newdim), rec['rmse']))

    # Create output and classify
    classes = classifier.classes_
    classified = np.zeros(rec.shape[0], dtype=[
        ('class', 'u2'),
        ('class_proba', 'float32', classes.size)
    ])
    classified['class'] = classifier.predict(X)
    classified['class_proba'] = classifier.predict_proba(X)

    # Replace with new classification if exists, or add by merging
    if ('class' in rec.dtype.names and 'class_proba' in rec.dtype.names and
            rec['class_proba'].shape[1] == classes.size):
        rec['class'] = classified['class']
        rec['class_proba'] = classified['class_proba']
    else:
        # Drop incompatible classified results if needed
        # e.g., if the number of classes changed
        if 'class' in rec.dtype.names and 'class_proba' in rec.dtype.names:
            rec = nprfn.drop_fields(rec, ['class', 'class_proba'])
        rec = nprfn.merge_arrays((rec, classified), flatten=True)

    # Create dict for re-saving `npz` file (only way to append)
    out = {}
    for k, v in z.iteritems():
        out[k] = v
    out['classes'] = classes
    out['record'] = rec

    np.savez(filename, **out)


# Main and parsing o/f arguments
def parse_args(args):
    """ Returns dictionary of parsed and validated command arguments

    Args:
      args (dict): Arguments from user

    Returns:
      dict: Parsed and validated arguments

    """
    parsed_args = {}
    # Required args
    parsed_args['config_file'] = args['<config_file>']

    parsed_args['algo'] = args['<trained_algorithm>']

    try:
        job_number = int(args['<job_number>'])
    except:
        print('Error - <job_number> must be an integer greater than 0')
        sys.exit(1)
    if job_number <= 0:
        print('Error - <job_number> cannot be less than or equal to 0')
        sys.exit(1)
    parsed_args['job_number'] = job_number

    try:
        parsed_args['total_jobs'] = int(args['<total_jobs>'])
    except:
        print('Error - <total_jobs> must be an integer')
        sys.exit(1)
    if parsed_args['job_number'] > parsed_args['total_jobs']:
        print('Error - <job_number> must be less than or equal to total jobs')
        sys.exit(1)

    parsed_args['resume'] = args['--resume']

    return parsed_args


def main(args):
    """ Classify dataset """
    # Parse config and file data
    dataset_config, yatsm_config = parse_config_file(args['config_file'])

    # Get some attributes about the dataset
    dates, images = csvfile_to_dataset(
        dataset_config['input_file'],
        date_format=dataset_config['date_format']
    )
    nrow, _, _, _ = get_image_attribute(images[0])

    # Read in the saved classification result
    try:
        _ = open(args['algo'])
    except:
        logger.error('Could not open pickled classifier')
        sys.exit(1)

    classifier = joblib.load(args['algo'])

    # Split into lines and classify
    job_lines = calculate_lines(args['job_number'] - 1, args['total_jobs'],
                                nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    start_time = time.time()
    logger.info('Starting to run lines')
    for job_line in job_lines:

        filename = get_output_name(dataset_config, job_line)
        if not os.path.exists(filename):
            logger.warning('No model result found for line {l} '
                           '(file {f})'.format(l=job_line, f=filename))
            pass

        if args['resume'] and try_resume(filename):
            logger.debug('Already processed line {l}'.format(l=job_line))
            continue

        logger.debug('Classifying line {l}'.format(l=job_line))
        classify_line(filename, classifier)

    logger.debug('Completed {n} lines in {m} minutes'.format(
        n=len(job_lines),
        m=round((time.time() - start_time) / 60.0, 2))
    )


if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)

    args = parse_args(args)
    main(args)
