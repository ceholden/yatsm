""" Assorted preprocessing utilities written in Cython for performance
"""
import numpy as np

cimport numpy as cnp
cimport cython

ctypedef fused np_int_1:
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t

ctypedef fused np_int_2:
    cnp.int16_t
    cnp.int32_t
    cnp.int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef cnp.ndarray[cnp.uint8_t, ndim=1] get_valid_mask(
        np_int_1[:, :] array,
        np_int_2[:] mins,
        np_int_2[:] maxes):
    """ Return mask for values of array within band by band min and max ranges

    Benchmark showed ~10x speed increase over pure NumPy solution (based on
    using np.all over multiple min < band < max logics).

    Args:
      array (np.ndarray): 2D array (# band x # observations, np.int32) to mask
      mins (np.ndarray): 1D array (# band, np.int32) of minimum range bound
      maxes (np.ndarray): 1D array (# band, np.int32) of maximum range bound

    Returns:
      np.ndarray: 1D (# observations, np.uint8) array mask with 1's for
        observations falling within the range and 0's for observations outside
        the specified range

    """
    cdef Py_ssize_t cols = array.shape[0]
    cdef Py_ssize_t rows = array.shape[1]
    cdef Py_ssize_t row, col
    cdef valid_mask = np.ones(rows, dtype=np.uint8)
    cdef cnp.uint8_t[:] valid_mask_view = valid_mask

    for row in range(rows):
        for col in range(cols):
            if array[col, row] <= mins[col] or array[col, row] >= maxes[col]:
                valid_mask_view[row] = 0
                break

    return valid_mask
