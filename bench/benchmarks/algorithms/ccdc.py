import inspect
import os

import numpy as np
import sklearn.linear_model

from yatsm.algorithms import CCDCesque

n = 50


# Hack for goof up in API previous to v0.6.0
def version_kwargs(d):
    """ Fix API calls for kwargs dict ``d`` that should have key ``estimator``
    """
    ccdc_args = inspect.getargspec(CCDCesque.__init__).args
    if 'estimator' in ccdc_args:
        return d
    elif 'lm' in ccdc_args:
        new_key, old_key = 'lm', 'estimator'
        d[new_key] = d.pop(old_key)
        return d
    else:
        raise KeyError('Neither "lm" nor "estimator" are keys in '
                       'CCDCesque.__init__')


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
            'estimator': sklearn.linear_model.Lasso(alpha=[20]),
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
        kwargs = version_kwargs(setup['kwargs'])
        for i in range(n):
            model = CCDCesque(**kwargs)
            model.fit(setup['X'], setup['Y'], setup['dates'])

    def time_ccdcesque2(self, setup):
        """ Bench with remove_noise turned on
        """
        kwargs = version_kwargs(setup['kwargs'])
        kwargs.update({'remove_noise': True})
        for i in range(n):
            model = CCDCesque(**kwargs)
            model.fit(setup['X'], setup['Y'], setup['dates'])

    def time_ccdcesque3(self, setup):
        """ Bench with remove_noise, dynamic_rmse turned on
        """
        kwargs = version_kwargs(setup['kwargs'])
        kwargs.update({'remove_noise': True,
                       'dynamic_rmse': True})
        for i in range(n):
            model = CCDCesque(**kwargs)
            model.fit(setup['X'], setup['Y'], setup['dates'])

    def time_ccdcesque4(self, setup):
        """ Bench with remove_noise, dynamic_rmse, slope_test turned on
        """
        kwargs = version_kwargs(setup['kwargs'])
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

    timeout = 360

    def setup_cache(self):
        dat = np.load(self.example_data)
        X = dat['X']
        Y = dat['Y']
        dates = dat['dates']

        kwargs = {
            'test_indices': np.array([2, 3, 4, 5]),
            'estimator': sklearn.linear_model.Lasso(alpha=[20]),
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
        kwargs = version_kwargs(setup['kwargs'])
        model = CCDCesque(**kwargs)
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])

    def time_ccdcesque2(self, setup):
        """ Bench with remove_noise turned on
        """
        kwargs = version_kwargs(setup['kwargs'])
        kwargs.update({'remove_noise': True})
        model = CCDCesque(**kwargs)
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])

    def time_ccdcesque3(self, setup):
        """ Bench with remove_noise, dynamic_rmse turned on
        """
        kwargs = version_kwargs(setup['kwargs'])
        kwargs.update({'remove_noise': True,
                       'dynamic_rmse': True})
        model = CCDCesque(**kwargs)
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])

    def time_ccdcesque4(self, setup):
        """ Bench with remove_noise, dynamic_rmse, slope_test turned on
        """
        kwargs = version_kwargs(setup['kwargs'])
        kwargs.update({'remove_noise': True,
                       'dynamic_rmse': True,
                       'slope_test': True})
        model = CCDCesque(**kwargs)
        for col in range(setup['Y'].shape[-1]):
            _Y, _X, _dates = setup['Y'][..., col], setup['X'], setup['dates']
            mask = np.in1d(_Y[-1, :], [0, 1])
            model.fit(_X[mask, :], _Y[:, mask], _dates[mask])
