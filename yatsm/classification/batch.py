""" Helper functions for running classifications
"""
import logging

import numpy as np
import numpy.lib.recfunctions as nprfn

logger = logging.getLogger(__name__)


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
    for k, v in z.items():
        out[k] = v
    out['classes'] = classes
    out['record'] = rec

    np.savez(filename, **out)
