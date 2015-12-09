import os

import numpy as np
import pytest

here = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def unmasked_ts(request):
    """ Return dict of an unmasked example timeseries

    Dict contains:
        dates: ordinal dates
        Y: observations of 7 bands + Fmask
        X: design matrix
        design_str: Patsy design specification
        design_dict: Patsy column name indices for design matrix X

    Mask based on Fmask values (retain 0 and 1) and optical data min/max
    of 0 to 10,000.
    """
    f = os.path.join(here, 'data', 'example_timeseries.npz')
    return np.load(f)


@pytest.fixture(scope='session')
def masked_ts(request):
    """ Return dict of a masked example timeseries

    Dict contains:
        dates: ordinal dates
        Y: observations of 7 bands + Fmask
        X: design matrix
        design_str: Patsy design specification
        design_dict: Patsy column name indices for design matrix X

    Mask based on Fmask values (retain 0 and 1) and optical data min/max
    of 0 to 10,000.
    """
    f = os.path.join(here, 'data', 'example_timeseries_masked.npz')
    return np.load(f)
