#!/usr/bin/env python
""" Open results and add some arbitrary classification attributes
"""
import os

import numpy as np
import numpy.lib.recfunctions as nprfn
import six

here = os.path.dirname(__file__)

out_dir = os.path.join(here, 'YATSM_classified')
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

results = [os.path.join(here, 'YATSM', 'yatsm_r%i.npz' % i) for i in range(5)]
classes = np.arange(1, 6)

for i, res in enumerate(results):
    # Load
    z = np.load(res)
    # Join records
    rec = z['record']
    classified = np.zeros(rec.shape[0], dtype=[
        ('class', 'u2'),
        ('class_proba', 'float32', classes.size)
    ])
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
    # Add in classification values
    rec['class'] = i
    rec['class_proba'] = 0.05
    rec['class_proba'][:, i] = 0.8
    assert np.all(rec['class_proba'].sum(axis=1) == 1.0), 'Bad proba calc'
    # Formulate output
    out = {}
    for k, v in six.iteritems(z):
        out[k] = v
    out['classes'] = classes
    out['record'] = rec
    # Save
    np.savez_compressed(os.path.join(out_dir, os.path.basename(res)), **out)
