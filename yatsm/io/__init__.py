""" YATSM IO module

.. todo::

   Include result file IO abstraction (:issue:`69`)

Contents:

    * :mod:`._util`: Collection of helper functions that ease common
      filesystem operations
"""
from ._util import mkdir_p


__all__ = [
    'mkdir_p'
]
