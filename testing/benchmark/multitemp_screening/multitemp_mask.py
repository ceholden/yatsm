""" Benchmark speed of Cython vs Python multi-temporal masking """
from __future__ import print_function, division
import os
import sys
import timeit

import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, 'yatsm'))

#from yatsm import multitemp_mask
#from cyatsm import multitemp_mask as cy_multitemp_mask


def test_setup():
    x = np.load(os.path.join(root, 'examples/sample_x.npy'))
    Y = np.load(os.path.join(root, 'examples/px96_py91_Y.npy'))

    mask = (Y[7, :] <= 1)

    x = np.array([_x.toordinal() for _x in x[mask]])
    Y = Y[:7, mask]

    i = np.arange(0, 24)
    n_year = (x[i[-1]] - x[i[0]]) / 365.25

    return x, Y, i, n_year


def py_test(x, Y, i, n_year):
    from yatsm import multitemp_mask
    return multitemp_mask(x[i], Y[:, i], n_year)


def cy_test(x, Y, i, n_year):
    from cyatsm import multitemp_mask
    return multitemp_mask(x[i], Y[:, i], n_year)


if __name__ == '__main__':
    n = 1000

    print('Python:')
    print(timeit.timeit('mtmask = py_test(x, Y, i, n_year)',
                        setup=('from __main__ import py_test, test_setup; '
                               'x, Y, i, n_year = test_setup()'),
                        number=n))

    print('Cython:')
    print(timeit.timeit('mtmask = cy_test(x, Y, i, n_year)',
                        setup=('from __main__ import cy_test, test_setup; '
                               'x, Y, i, n_year = test_setup()'),
                        number=n))

#    from IPython.core.debugger import Pdb
#    Pdb().set_trace()
