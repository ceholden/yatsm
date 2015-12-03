""" Tests for yatsm.phenology.longtermmean
"""
import numpy as np

import yatsm.phenology.longtermmean as ltm


def test_scale_EVI(df):
    evi_norm = ltm.scale_EVI(df['evi'].values, df['prd'].values)
    np.testing.assert_almost_equal(evi_norm.max(), 1.69351, decimal=5,
                                   err_msg='Scaled EVI max is not correct')
    np.testing.assert_almost_equal(evi_norm.min(), -0.2385107, decimal=5,
                                   err_msg='Scaled EVI min is not correct')
