""" Calculations one might want to perform in a preprocessing pipeline
"""
import logging

from .._task_validation import eager_task, requires, outputs

logger = logging.getLogger(__name__)


@eager_task
@requires(data=[str, str])
@outputs(data=[str])
def norm_diff(work, require, output, **config):
    """ Calculate a normalized difference of two bands

    Args:
        work: Dataset to operate on
        require (dict[str, list[str]]): Labels for the requirements of this
            calculation
        output (dict[str, list[str]]): Labels for the result of this
            calculation

    Returns:
        dict: Input ``work`` dictionary with the calculation added according
            to user specification of ``output``
    """
    one, two = require['data']
    out = output['data'][0]

    work['data'][out] = ((work['data'][one] - work['data'][two]) /
                         (work['data'][one] + work['data'][two]))

    return work
