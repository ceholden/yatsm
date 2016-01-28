#!/usr/bin/env python
""" Test for prediction and coefficient normalization calculation speeds

Test looping over indices found and using simple dot products, versus
more complicated numpy functions.

"""
from __future__ import print_function

import timeit

import numpy as np


def setup():
    # Setup
    rec = np.load('yatsm_r200.npz')['record']

    d = 730508
    index = np.where((rec['start'] < d) & (rec['end'] > d))[0]
    # index = np.array([0, 1])

    ncol = np.unique(rec['px']).size
    ncoef = rec['coef'].shape[1]
    nband = rec['coef'].shape[2]

    X = np.array([1, d,
                  np.cos(2 * np.pi / 365.25 * d),
                  np.sin(2 * np.pi / 365.25 * d)])

    return (rec, index, X, ncol, ncoef, nband)


def predict_looped(rec, index, X, shape):
    # Variable to save into
    line = np.zeros(shape)
    bands = np.arange(shape[1])

    for i in index:
        for i_b, b in enumerate(bands):
            # Calculate predicted image
            line[rec['px'][i], i_b] = \
                np.dot(rec['coef'][i][:, b], X)

    return line


def predict_tensor(rec, index, X, shape):
    # Variable to save into
    line = np.zeros(shape)
    bands = np.arange(shape[1])

    line[rec['px'][index]] = np.tensordot(rec['coef'][index], X, axes=(1, 0))

    return line


def coef_loop(rec, index, shape):
    # Variable to save into
    line = np.zeros(shape)
    coefs = np.arange(shape[1])
    bands = np.arange(shape[2])

    for i in index:
        # Normalize intercept to mid-point in time segment
        rec['coef'][i][0, :] = rec['coef'][i][0, :] + \
            (rec['start'][i] + rec['end'][i]) / 2.0 * rec['coef'][i][1, :]
        # Extract coefficients
        line[rec['px'][i]] = rec['coef'][i][coefs, :][:, bands]

    return line


def coef_dot(rec, index, shape):
    # Variable to save into
    line = np.zeros(shape)
    coefs = np.arange(shape[1])
    bands = np.arange(shape[2])

    # Normalize intercept to mid-point in time segment
    rec['coef'][index, 0, :] += \
        ((rec['start'][index] + rec['end'][index]) / 2.0)[:, None] * \
        rec['coef'][index, 1, :]
    # Extract coefficients
    line[rec['px'][index]] = rec['coef'][index][:, coefs, :][:, :, bands]

    return line


if __name__ == '__main__':
    n = 1000

    _setup = ('from __main__ import setup, \
               predict_looped, predict_tensor, coef_loop, coef_dot; '
              'rec, index, X, ncol, ncoef, nband = setup()')
    rec, index, X, ncol, ncoef, nband = setup()

    print('PREDICTION:')
    print('Loop version:')
    speed_1 = timeit.timeit(
        'result_1 = predict_looped(rec, index, X, (ncol, nband))',
        setup=_setup, number=n)
    print(speed_1)
    result_1 = predict_looped(rec, index, X, (ncol, nband))

    print('numpy.tensordot')
    speed_2 = timeit.timeit(
        'result_2 = predict_tensor(rec, index, X, (ncol, nband))',
        setup=_setup, number=n)
    print(speed_2)
    result_2 = predict_tensor(rec, index, X, (ncol, nband))

    np.testing.assert_allclose(result_1, result_2, rtol=1e-5)
    print('tensordot speed: {p}% faster\n'.format(
        p=round(speed_1 / speed_2 * 100, 3)))

    print('COEFICIENT:')
    print('Loop version:')
    speed_1 = timeit.timeit(
        'coef_loop(rec, index, (ncol, ncoef, nband))',
        setup=_setup, number=n)
    print(speed_1)
    result_1 = coef_loop(rec, index, (ncol, ncoef, nband))

    rec, index, X, ncol, ncoef, nband = setup()

    print('numpy.dot')
    speed_2 = timeit.timeit(
        'coef_dot(rec, index, (ncol, ncoef, nband))',
        setup=_setup, number=n)
    print(speed_2)
    result_2 = coef_dot(rec, index, (ncol, ncoef, nband))

    np.testing.assert_allclose(result_1, result_2, rtol=1e-3)
    print('dot product speed: {p}% faster\n'.format(
        p=round(speed_1 / speed_2 * 100, 3)))

    print('NumPy version: %s' % np.__version__)
