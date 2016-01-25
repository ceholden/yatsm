""" Command line interface for classifying YATSM algorithm output
"""
from __future__ import division, print_function

import logging
import os
import time

import click
import numpy as np
import numpy.lib.recfunctions as nprfn
import six
from sklearn.externals import joblib

from . import options
from ..config_parser import parse_config_file
from ..utils import distribute_jobs, get_output_name, csvfile_to_dataframe
from ..io import get_image_attribute

logger = logging.getLogger('yatsm')


@click.command(short_help='Classify entire images using trained algorithm')
@options.arg_config_file
@click.argument('algo', metavar='<trained algorithm>',
                type=click.Path(readable=True, resolve_path=True))
@options.arg_job_number
@options.arg_total_jobs
@click.option('--resume', is_flag=True,
              help="Resume classification (don't overwrite)")
@click.pass_context
def classify(ctx, config, algo, job_number, total_jobs, resume):
    cfg = parse_config_file(config)

    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              cfg['dataset']['date_format'])
    nrow = get_image_attribute(df['filename'][0])[0]

    classifier = joblib.load(algo)

    # Split into lines and classify
    job_lines = distribute_jobs(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    start_time = time.time()
    logger.info('Starting to run lines')
    for job_line in job_lines:
        filename = get_output_name(cfg['dataset'], job_line)
        if not os.path.exists(filename):
            logger.warning('No model result found for line {l} '
                           '(file {f})'.format(l=job_line, f=filename))
            pass

        if resume and try_resume(filename):
            logger.debug('Already processed line {l}'.format(l=job_line))
            continue

        logger.debug('Classifying line {l}'.format(l=job_line))
        classify_line(filename, classifier)

    logger.debug('Completed {n} lines in {m} minutes'.format(
        n=len(job_lines),
        m=round((time.time() - start_time) / 60.0, 2))
    )


def try_resume(filename):
    """ Return True/False if dataset has already been classified

    Args:
        filename (str): filename of the result to be checked

    Returns:
        bool: If the `npz` file exists and contains a file 'class', this test
            will return True, else False.

    """
    try:
        z = np.load(filename)
    except:
        return False

    if not z['record'].dtype or 'class' not in z['record'].dtype.names:
        return False

    return True


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
    coef = rec['coef'].copy()  # copy so we don't transform npz coef
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
    for k, v in six.iteritems(z):
        out[k] = v
    out['classes'] = classes
    out['record'] = rec

    np.savez(filename, **out)
