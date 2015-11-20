""" Tests for yatsm.phenology.longtermmean
"""
import os

import numpy as np
import pytest
import yatsm.phenology.longtermmean as ltm

here = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def df(request):
    """ Return a Pandas df of example EVI timeseries """
    import pandas as pd
    df = pd.read_csv(os.path.join(here, 'data', 'evi_test.csv'))
    df.columns = ['yr', 'doy', 'prd', 'evi']

    # Mask invalid EVI data
    mask = np.where((df['evi'] >= 0) & (df['evi'] <= 1))[0]
    df = df.ix[mask]

    return df


def test_scale_EVI(df):
    evi_norm = ltm.scale_EVI(df['evi'].values, df['prd'].values)
    np.testing.assert_almost_equal(evi_norm.max(), 1.69351, decimal=5,
                                   err_msg='Scaled EVI max is not correct')
    np.testing.assert_almost_equal(evi_norm.min(), -0.2385107, decimal=5,
                                   err_msg='Scaled EVI min is not correct')
