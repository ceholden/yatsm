import os

import numpy as np
import sklearn.linear_model

from yatsm import __version__ as yatsm_version
from yatsm.algorithms import CCDCesque

n = 50
# Hack for goof up in API previous to v0.6.0
if yatsm_version.split('.')[1] == '5':
    est = 'lm'
else:
    est = 'estimator'


class CCDCesquePixel263(object):
    """ Benchmark CCDC-esque algorithm on a single pixel with 263 observations
    """
    example_data = os.path.join(
        os.path.dirname(__file__),
        '../../../tests/algorithms/data/example_timeseries_masked.npz')

    def setup_cache(self):
        dat = np.load(self.example_data)
        X = dat['X']
        Y = dat['Y']
        dates = dat['dates']

        kwargs = {
            'test_indices': np.array([2, 3, 4, 5]),
            est: sklearn.linear_model.Lasso(alpha=[20]),
            'consecutive': 5,
            'threshold': 4,
            'min_obs': 24,
            'min_rmse': 100,
            'retrain_time': 365.25,
            'screening': 'RLM',
            'screening_crit': 400.0,
            'green_band': 1,
            'swir1_band': 4,
            'remove_noise': False,
            'dynamic_rmse': False,
            'slope_test': False,
            'idx_slope': 1
        }
        return {'X': X, 'Y': Y, 'dates': dates, 'kwargs': kwargs}

    def time_ccdcesque1(self, setup):
        """ Bench with 'defaults' defined in setup with most tests turned off
        """
        for i in range(n):
            model = CCDCesque(**setup['kwargs'])
            model.fit(setup['X'], setup['Y'], setup['dates'])

    def time_ccdcesque2(self, setup):
        """ Bench with remove_noise turned on
        """
        kwargs = setup['kwargs']
        kwargs.update({'remove_noise': True})
        for i in range(n):
            model = CCDCesque(**kwargs)
            model.fit(setup['X'], setup['Y'], setup['dates'])

    def time_ccdcesque3(self, setup):
        """ Bench with remove_noise, dynamic_rmse turned on
        """
        kwargs = setup['kwargs']
        kwargs.update({'remove_noise': True,
                       'dynamic_rmse': True})
        for i in range(n):
            model = CCDCesque(**kwargs)
            model.fit(setup['X'], setup['Y'], setup['dates'])

    def time_ccdcesque4(self, setup):
        """ Bench with remove_noise, dynamic_rmse, slope_test turned on
        """
        kwargs = setup['kwargs']
        kwargs.update({'remove_noise': True,
                       'dynamic_rmse': True,
                       'slope_test': True})
        for i in range(n):
            model = CCDCesque(**kwargs)
            model.fit(setup['X'], setup['Y'], setup['dates'])


class CCDCesqueLine(object):
    """ Benchmark CCDC-esque algorithm on a line with TODO observations
    """
    example_data = os.path.join(
        os.path.dirname(__file__),
        '../../../tests/data/p013r030_r50_n423_b8.npz')

    def setup_cache(self):
        dat = np.load(self.example_data)
        X = dat['X']
        Y = dat['Y']
        dates = dat['dates']

        kwargs = {
            'test_indices': np.array([2, 3, 4, 5]),
            est: sklearn.linear_model.Lasso(alpha=[20]),
            'consecutive': 5,
            'threshold': 4,
            'min_obs': 24,
            'min_rmse': 100,
            'retrain_time': 365.25,
            'screening': 'RLM',
            'screening_crit': 400.0,
            'green_band': 1,
            'swir1_band': 4,
            'remove_noise': False,
            'dynamic_rmse': False,
            'slope_test': False,
            'idx_slope': 1
        }
        return {'X': X, 'Y': Y, 'dates': dates, 'kwargs': kwargs}

    def time_ccdcesque1(self, setup):
        """ Bench with 'defaults' defined in setup with most tests turned off
        """
        model = CCDCesque(**setup['kwargs'])
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])

    def time_ccdcesque2(self, setup):
        """ Bench with remove_noise turned on
        """
        model = CCDCesque(**setup['kwargs'])
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])

    def time_ccdcesque3(self, setup):
        """ Bench with remove_noise, dynamic_rmse turned on
        """
        model = CCDCesque(**setup['kwargs'])
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])

    def time_ccdcesque4(self, setup):
        """ Bench with remove_noise, dynamic_rmse, slope_test turned on
        """
        model = CCDCesque(**setup['kwargs'])
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])
