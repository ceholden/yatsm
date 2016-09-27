""" Functions for running various processing task in a pipeline
"""
from .preprocess import norm_diff
from .change import pixel_CCDCesque


# TODO: allow 3rd party tasks via an entry point
PIPELINE_TASKS = {
    # DATA MANIPULATION
    'norm_diff': norm_diff,
    # CHANGE DETECTION
    'pixel_CCDCesque': pixel_CCDCesque
}
