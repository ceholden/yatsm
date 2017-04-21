""" Calculations one might want to perform in a preprocessing pipeline
"""
import logging

import numpy as np
import patsy
import xarray as xr

from yatsm.pipeline.tasks._validation import (eager_task, requires, outputs,
                                              version)
from yatsm.regression.transforms import harm  # NOQA

logger = logging.getLogger(__name__)


@version('norm_diff:1.0.0')
@eager_task
@requires(data=[str, str])
@outputs(data=[str])
def norm_diff(pipe, require, output, config=None):
    """ Calculate a normalized difference of two bands

    Args:
        pipe (yatsm.pipeline.Pipe): Piped data to operate on
        require (dict[str, list[str]]): Labels for the requirements of this
            calculation
        output (dict[str, list[str]]): Label for the result of this
            calculation

    Returns:
        yatsm.pipeline.Pipe: Piped output
    """
    one, two = require['data']
    out = output['data'][0]

    pipe.data[out] = ((pipe.data[one] - pipe.data[two]) /
                      (pipe.data[one] + pipe.data[two]))

    return pipe


@version('dmatrix:1.0.0')
@eager_task
@requires(data=[])
@outputs(data=[str])
def dmatrix(pipe, require, output, config=None):
    """ Create a design matrix from Patsy/R style design strings

    Args:
        pipe (yatsm.pipeline.Pipe): Piped data to operate on
        require (dict[str, list[str]]): Labels for the requirements of this
            calculation
        output (dict[str, list[str]]): Label for the result of this
            calculation
        config

    Returns:
        yatsm.pipeline.Pipe: Piped output
    """
    design = config['design']
    ds = pipe.data

    if '~' in design:
        X = patsy.dmatrices(design, ds)
    elif design.strip() == '1':
        X = np.ones((ds['time'].size, 1))
        X = patsy.DesignMatrix(X, design_info=patsy.DesignInfo(['Intercept']))
    else:
        X = patsy.dmatrix(design, ds)

    coords = (ds['time'], X.design_info.column_names)
    dims = ('time', 'terms')
    pipe.data[output['data'][0]] = xr.DataArray(X, coords, dims)

    return pipe
