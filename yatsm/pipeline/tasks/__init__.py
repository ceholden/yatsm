""" Functions for running various processing task in a pipeline

.. todo::

    Allow specification of pipeline tasks using entry points

"""
from .preprocess import norm_diff
from .change import pixel_CCDCesque


#: dict: Tasks that generate segments, usually through
#        some kind of change detection process
SEGMENT_TASKS = {
    # CHANGE
    'pixel_CCDCesque': pixel_CCDCesque,
}


PIPELINE_TASKS = {
    # DATA MANIPULATION
    'norm_diff': norm_diff,
}
PIPELINE_TASKS.update(SEGMENT_TASKS)
