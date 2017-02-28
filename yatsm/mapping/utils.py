""" Utilities for turning YATSM record results into maps

Also stores definitions for model QA/QC values
"""
import logging

logger = logging.getLogger('yatsm')

# QA/QC values for segment types
MODEL_QA_QC = {
    'INTERSECT': 3,
    'AFTER': 2,
    'BEFORE': 1
}
