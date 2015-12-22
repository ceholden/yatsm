""" Baseclass for benchmarking against an example timeseries
"""
import os

import numpy as np


class PixelTimeseries(object):
    """ Setup example timeseries of a pixel

    Attributes:
        dat (np.lib.npyio.NpzFile): saved timeseries
        X (np.ndarray): n_obs x n_features design matrix
        Y (np.ndarray): n_series x n_obs independent variable
        dates (np.ndarray): n_obs array of ordinal dates for each ``Y``
    """
    example_data = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..', 'tests', 'algorithms', 'data',
        'example_timeseries_masked.npz')

    def setup_cache(self):
        self.dat = np.load(self.example_data)
        self.X = self.dat['X']
        self.Y = self.dat['Y']
        self.dates = self.dat['dates']
        return {'X': self.X, 'Y': self.Y, 'dates': self.dates}
