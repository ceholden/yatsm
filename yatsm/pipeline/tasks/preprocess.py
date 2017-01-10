""" Calculations one might want to perform in a preprocessing pipeline
"""
import logging

import numpy as np
import patsy
import xarray as xr

from .._task_validation import eager_task, requires, outputs
from ...regression.transforms import harm

logger = logging.getLogger(__name__)


@eager_task
@requires(data=[str, str])
@outputs(data=[str])
def norm_diff(work, require, output, config=None):
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


@eager_task
@requires(data=[])
@outputs(data=[str])
def dmatrix(work, require, output, config=None):
    """ Create a design matrix from Patsy/R style design strings
    """
    design = config['design']
    ds = work['data']

    if '~' in design:
        X = patsy.dmatrices(design, ds)
    elif design.strip() == '1':
        X = np.ones((ds['time'].size, 1))
        X = patsy.DesignMatrix(X, design_info=patsy.DesignInfo(['Intercept']))
    else:
        X = patsy.dmatrix(design, ds)

    coords = (ds['time'], X.design_info.column_names)
    dims = ('time', 'terms')
    work['data'][output['data'][0]] = xr.DataArray(X, coords, dims)

    return work
