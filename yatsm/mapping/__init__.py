""" Module for making map products from YATSM results

Contains functions used in "map" command line interface script.
"""
from .core import result_map

# QA/QC values for segment types
MODEL_QA_QC = {
    'INTERSECT': 3,
    'AFTER': 2,
    'BEFORE': 1
}


__all__ = [
    'MODEL_QA_QC',
    'result_map',
]
