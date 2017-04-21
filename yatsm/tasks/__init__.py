""" Functions for running various processing task in a pipeline

.. todo::

    Document how to create new tasks (e.g., how to use tools inside
    :ref:`yatsm.pipeline._validation`)

.. todo::

    Document how to add new tasks via entry points

"""
from .change import pixel_CCDCesque
from .preprocess import dmatrix, norm_diff
from .stash import sklearn_dump, sklearn_load


__all__ = [
    # change
    'pixel_CCDCesque',
    # preprocess
    'dmatrix',
    'norm_diff',
    # stash
    'sklearn_dump',
    'sklearn_load'
]
