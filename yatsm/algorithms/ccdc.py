from __future__ import print_function, division

import logging
import sys

import numpy as np

import sklearn.linear_model

from .yatsm import YATSM
from ..accel import try_jit
from ..errors import TSLengthException
from ..masking import smooth_mask, multitemp_mask
from ..regression.diagnostics import rmse

# Setup
logger = logging.getLogger('yatsm_algo')


@try_jit(nopython=True)
def _monitor_calc_scores(X, Y, here, scores, predictions, rmse,
                         test_indices, min_rmse):
    """ Calculate monitoring period scaled residuals
    """
    for i in range(scores.shape[1]):
        for i_b, b in enumerate(test_indices):
            scores[i_b, i] = (
                (Y[b, here + i] - predictions[i_b, i]) /
                max(min_rmse, rmse[i_b])
            )


class CCDCesque(YATSM):
    """Initialize a CCDC-like model for data X (spectra) and Y (dates)

    An unofficial and unvalidated port of the Continuous Change Detection and
    Classification (CCDC) algorithm by Zhu and Woodcock, 2014.

    Args:
        test_indices (np.ndarray, optional): Test for changes with these
            indices of Y. If not provided, all series in `Y` will be used as
            test indices
        estimator (sklearn compatible predictor): prediction model from
            scikit-learn used to fit and predict timeseries
        consecutive (int, optional): Consecutive observations to trigger change
        threshold (float): Test statistic threshold for change
        min_obs (int): Minimum observations in model
        min_rmse (float): Minimum RMSE for models during testing
        retrain_time (float): Number of days between model fit updates during
            monitoring period
        screening (str): Style of prescreening of the timeseries for noise.
            Options are 'RLM' or 'LOWESS' (default: RLM)
        screening_crit (float, optional): critical value for multitemporal
            noise screening (default: 400.0)
        remove_noise (bool, optional): Remove observation if change is not
            detected but first observation is above threshold (if it looks like
            noise) (default: True)
        green_band (int, optional): Index of green band in Y for
            multitemporal masking (default: 1)
        swir1_band (int, optional): Index of first SWIR band in Y for
            multitemporal masking (default: 4)
        dynamic_rmse (bool, optional): Vary RMSE as a function of day of year
            (default: False)
        slope_test (float or bool, optional): Use an additional slope test to
            assess the suitability of the training period. A value of True
            enables the test and uses the `threshold` parameter as the test
            criterion. False turns off the test or a float value enables the
            test but overrides the test criterion threshold. (default: False)
        idx_slope (int): if ``slope_test`` is enabled, provide index of ``X``
            containing slope term (default: 1)

    """

    ndays = 365.25

    def __init__(self,
                 test_indices=None,
                 estimator=sklearn.linear_model.Lasso(alpha=20),
                 consecutive=5, threshold=2.56, min_obs=None, min_rmse=None,
                 retrain_time=365.25, screening='RLM', screening_crit=400.0,
                 remove_noise=True, green_band=1, swir1_band=4,
                 dynamic_rmse=False, slope_test=False, idx_slope=1):
        # Parent sets up test_indices and lm
        super(CCDCesque, self).__init__(test_indices, estimator)

        # Store model hyperparameters
        self.consecutive = consecutive
        self.threshold = threshold
        self.min_obs = min_obs or 16
        self.min_rmse = min_rmse or sys.float_info.min
        self.retrain_time = retrain_time

        # Define screening method according to type
        if screening == 'RLM':
            self.screen_timeseries = self._screen_timeseries_RLM
            logger.debug('Using RLM for screening')
        elif screening == 'LOWESS':
            self.screen_timeseries = self._screen_timeseries_LOWESS
            logger.debug('Using LOWESS for screening')
        else:
            raise TypeError('Unknown screening type %s' % screening)

        self.screening_crit = screening_crit
        self.remove_noise = remove_noise
        self.green_band = green_band
        self.swir1_band = swir1_band
        self.slope_test = slope_test
        if self.slope_test is True:
            self.slope_test = threshold
        self.idx_slope = idx_slope

        if dynamic_rmse:
            self.get_rmse = self._get_dynamic_rmse
        else:
            self.get_rmse = self._get_model_rmse

    @property
    def record_template(self):
        """ Return a YATSM record template for features in X and series in Y

        Record template will set `px` and `py` if defined as class attributes.
        Otherwise `px` and `py` coordinates will default to 0.

        Returns:
            np.ndarray: NumPy structured array containing a template of a YATSM
                record

        """
        record_template = np.zeros(1, dtype=[
            ('start', 'i4'),
            ('end', 'i4'),
            ('break', 'i4'),
            ('coef', 'float32', (self.n_features, self.n_series)),
            ('rmse', 'float32', (self.n_series)),
            ('magnitude', 'float32', self.n_series),
            ('px', 'u2'),
            ('py', 'u2')
        ])
        record_template['px'] = getattr(self, 'px', 0)
        record_template['py'] = getattr(self, 'py', 0)

        return record_template

# HELPER PROPERTIES
    @property
    def span_time(self):
        """ Return time span (in days) between start and end of model """
        return abs(self.dates[self.here] - self.dates[self.start])

    @property
    def span_index(self):
        """ Return time span (in index) between start and end of model """
        return (self.here - self.start)

    @property
    def running(self):
        """ Determine if timeseries can run """
        return self.here < len(self.dates)

    @property
    def can_monitor(self):
        """ Determine if timeseries can monitor the future consecutive obs """
        return self.here < len(self.dates) - self.consecutive - 1

# MAIN LOOP
    def fit(self, X, Y, dates):
        """ Fit timeseries model

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (np.ndarray): ordinal dates for each observation in X/Y

        Returns:
            np.ndarray: NumPy structured array containing timeseries
                model attribute information

        """
        if len(dates) != X.shape[0] or len(dates) != Y.shape[1]:
            raise ValueError('X/Y/dates must have same number of observations')

        self.X = np.asarray(X, dtype=np.float64)
        self.Y = np.asarray(Y, dtype=np.float64)
        self.dates = dates
        self.n_features = X.shape[1]
        self.n_series = Y.shape[0]

        if not np.any(np.asarray(self.test_indices)):
            self.test_indices = np.arange(self.n_series)

        # Set or reset state variables
        self.reset()

        if len(dates) < self.here + self.consecutive:
            raise TSLengthException('Not enough observations (n = %s)' %
                                    len(dates))

        self.n_record = 0
        self.record = np.copy(self.record_template)

        while self.running:

            while not self.monitoring and self.can_monitor:
                self.train()
                self.here += 1

            while self.monitoring and self.can_monitor:
                # Update model if required
                self._update_model()
                # Perform monitoring check
                self.monitor()
                # Iterate forward
                self.here += 1

            self.here += 1

        # Update record for last model
        self.record[self.n_record]['start'] = self.dates[self.start]
        # Re-adjust end for consecutive, and for two ``self.here += 1`` calls
        offset = 1 + (1 if self.monitoring else 0)
        self.record[self.n_record]['end'] = self.dates[
            self.here - self.consecutive - offset]
        for i, m in enumerate(self.models):
            self.record[self.n_record]['coef'][:, i] = m.coef
            self.record[self.n_record]['rmse'][i] = m.rmse

        # If we ended without being able to monitor again, delete last model
        # since it will be empty
        # TODO: fit this time period with median
        if not self.monitoring:
            self.record = self.record[:-1]

        return self.record

    def reset(self):
        """ Reset state information required for model fittings
        """
        # Location information
        self.start = 0
        self.here = self.min_obs
        self._here = self.here
        self.trained_date = 0
        self.monitoring = False
        # Populate prediction models
        if len(self.models) == 0:
            self.models = np.array([sklearn.clone(self.estimator) for
                                    i in range(self.n_series)])
            for m in self.models:  # initialize additional attributes
                m.rmse = 0.0
                m.coef = np.zeros(self.X.shape[1])
        # Training period test calculations
        self.start_resid = np.zeros(len(self.test_indices))
        self.end_resid = np.zeros(len(self.test_indices))
        self.slope_resid = np.zeros(len(self.test_indices))
        # Monitoring period calculations
        self.predictions = np.zeros((len(self.test_indices), self.consecutive),
                                    dtype=np.float64)
        self.scores = np.zeros((len(self.test_indices), self.consecutive),
                               dtype=np.float64)

    def train(self):
        """ Train time series model if stability criteria are met

        Stability criteria (Equation 5 in Zhu and Woodcock, 2014) include a
        test on the change in reflectance over the training period (slope test)
        and a test on the magnitude of the residuals for the first and last
        observations in the training period. Training periods with large slopes
        can indicate that a disturbance process is still in progress. Large
        residuals on the first or last observations have high leverage on the
        estimated regression and should be excluded from the training period.

        1. Slope test:

        .. math::
            \\frac{1}{n}\sum\limits_{b\in B_{test}}\\frac{
                \left|\\beta_{slope,b}(t_{end}-t_{start})\\right|}
                {RMSE_b} > T_{crit}

        2. First and last residual tests:

        .. math::
            \\frac{1}{n}\sum\limits_{b\in B_{test}}\\frac{
                \left|\hat\\rho_{b,i=1} - \\rho_{b,i=1}\\right|}
                {RMSE_b} > T_{crit}

            \\frac{1}{n}\sum\limits_{b\in B_{test}}\\frac{
                \left|\hat\\rho_{b,i=N} - \\rho_{b,i=N}\\right|}
                {RMSE_b} > T_{crit}

        """
        # Test if we can train yet
        if self.span_time <= self.ndays or self.span_index < self.n_features:
            logger.debug('Could not train - moving forward')
            return

        # Check if screening was OK
        if not self.screen_timeseries():
            return

        # Test if we can still run after noise removal
        if self.here >= self._X.shape[0]:
            # logger.debug('Not enough observations to proceed after noise '
            #              'removal')
            # return
            raise TSLengthException('Not enough observations to proceed '
                                    'after noise removal')

        # After noise removal, try to fit models
        self._fit_models(self._X[self.start:self.here + 1, :],
                         self._Y[:, self.start:self.here + 1],
                         bands=self.test_indices)

        # Ensure first and last points aren't unusual
        for i, b in enumerate(self.test_indices):
            m = self.models[b]
            _rmse = max(self.min_rmse, m.rmse)
            self.start_resid[i] = (
                np.abs(self._Y[b, self.start] -
                       m.predict(self._X[self.start, :][None, :])) /
                _rmse)
            self.end_resid[i] = (
                np.abs(self._Y[b, self.here] -
                       m.predict(self._X[self.here, :][None, :])) /
                _rmse)
            self.slope_resid[i] = (
                np.abs(m.coef_[self.idx_slope] * (self.here - self.start)) /
                _rmse)

        test_start = np.linalg.norm(self.start_resid)
        test_end = np.linalg.norm(self.end_resid)
        test_slope = np.linalg.norm(self.slope_resid)
        if (test_start > self.threshold or test_end > self.threshold or
                (self.slope_test and test_slope > self.threshold)):
            logger.debug('Training period unstable')
            self.start += 1
            self.here = self._here
            return

        self.X = self._X
        self.Y = self._Y
        self.dates = self._dates

        logger.debug('Entering monitoring period')
        self.monitoring = True

    def monitor(self):
        """ Monitor for changes in time series """
        _rmse = self.get_rmse()

        for idx, model in enumerate(self.models[self.test_indices]):
            self.predictions[idx, :] = model.predict(
                self.X[self.here:self.here + self.consecutive, :])

        _monitor_calc_scores(self.X, self.Y, self.here,
                             self.scores,
                             self.predictions, _rmse,
                             self.test_indices,
                             self.min_rmse)

        # Check for scores above critical value
        mag = np.linalg.norm(self.scores, axis=0)

        if np.all(mag > self.threshold):
            logger.debug('CHANGE DETECTED')

            # Update record for last model
            self.record[self.n_record]['start'] = self.dates[self.start]
            self.record[self.n_record]['end'] = self.dates[self.here]
            self.record[self.n_record]['break'] = self.dates[self.here + 1]
            for i, m in enumerate(self.models):
                self.record[self.n_record]['coef'][:, i] = m.coef
                self.record[self.n_record]['rmse'][i] = m.rmse

            # Record magnitude of difference for tested indices
            self.record[self.n_record]['magnitude'][self.test_indices] = \
                np.mean(self.scores, axis=1)

            self.record = np.append(self.record, self.record_template)
            self.n_record += 1

            # Reset _X and _Y for re-training
            self._X = self.X
            self._Y = self.Y
            self.start = self.here + 1

            self.trained_date = 0
            self.monitoring = False

        elif mag[0] > self.threshold and self.remove_noise:
            # Masking way of deleting is faster than `np.delete`
            m = np.ones(self.X.shape[0], dtype=bool)
            m[self.here] = False
            self.X = self.X[m, :]
            self.Y = self.Y[:, m]
            self.dates = self.dates[m]
            self.here -= 1

# MODEL FITTING UTILITIES
    def _update_model(self):
        # Only train if enough time has past
        if (abs(self.dates[self.here] - self.trained_date) >
                self.retrain_time):
            logger.debug('Monitoring - retraining (%s days since last)' %
                         str(self.dates[self.here] - self.trained_date))

            # Fit timeseries models
            self._fit_models(self.X[self.start:self.here + 1, :],
                             self.Y[:, self.start:self.here + 1])

            self.trained_date = self.dates[self.here]

    def _fit_models(self, X, Y, bands=None):
        """ Fit timeseries models for `bands` within `Y` for a given `X`

        Updates or initializes fit for ``self.models``

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations) observation in the X design matrix
            bands (iterable): Subset of bands of `Y` to fit. If None are
                provided, fit all bands in Y

        """
        if bands is None:
            bands = np.arange(self.n_series)

        for b in bands:
            y = Y.take(b, axis=0)

            model = self.models[b]
            model.fit(X, y)

            # Add in RMSE calculation  # TODO: numba?
            model.rmse = rmse(y, model.predict(X))

            # Add intercept to intercept term of design matrix
            model.coef = model.coef_.copy()
            model.coef[0] += model.intercept_

# MULTITEMP SCREENING
    def _screen_timeseries_LOWESS(self, span=None):
        """ Screen entire dataset for noise before training using LOWESS

        Args:
            span (int, optional): span for LOWESS

        Returns:
            bool: True if timeseries is screened and we can train, else False

        """
        if not self.screened:
            span = span or self.consecutive * 2 + 1

            mask = smooth_mask(self.dates, self.Y, span,
                               crit=self.screening_crit,
                               green=self.green_band, swir1=self.swir1_band)

            # Apply mask to X and Y
            self.X = self.X[mask, :]
            self.Y = self.Y[:, mask]
            # Also apply to _X and _Y for training purposes
            self._X = self.X
            self._Y = self.Y

            self.screened = True

        return True

    def _screen_timeseries_RLM(self):
        """ Screen training period for noise with IRWLS RLM

        Returns:
            bool: True if timeseries is screened and we can train, else False

        """
        # Multitemporal noise removal
        mask = np.ones(self.X.shape[0], dtype=np.bool)
        index = np.arange(self.start, self.here + self.consecutive,
                          dtype=np.uint16)
        mask[index] = multitemp_mask(self.dates[index],
                                     self.Y[:, index],
                                     self.span_time / self.ndays,
                                     crit=self.screening_crit,
                                     green=self.green_band,
                                     swir1=self.swir1_band)

        # Check if there are enough observations for model with noise removed
        _span_index = mask[index][:-self.consecutive].sum()

        # Return if not enough observations
        if _span_index < self.min_obs:
            logger.debug('    multitemp masking - not enough obs')
            return False

        # There is enough observations in train period to fit - remove noise
        self._X = self.X[mask, :]
        self._Y = self.Y[:, mask]
        self._dates = self.dates[mask]

        # record our current position
        #   important for next iteration of noise removal
        self._here = self.here

        # Go forward after noise removal
        self.here = self.start + _span_index - 1

        if self.span_time < self.ndays:
            logger.debug('    multitemp masking - not enough time')
            self.here = self._here
            return False

        logger.debug('Updated "here"')

        return True

# RMSE CALCULATION
    def _get_model_rmse(self):
        """ Return the normal RMSE of each fitted model

        Returns:
          np.ndarray: NumPy array containing RMSE of each tested model

        """
        return np.array([m.rmse for m in self.models])[self.test_indices]

    def _get_dynamic_rmse(self):
        """ Return the dynamic RMSE for each model

        Dynamic RMSE refers to the Root Mean Squared Error calculated using
        `self.min_obs` number of observations closest in day of year to the
        observation `self.consecutive` steps into the future. Goal is to
        reduce false-positives during seasonal transitions (high variance in
        the signal) while decreasing omission during stable times of year.

        Returns:
          np.ndarray: NumPy array containing dynamic RMSE of each tested model

        """
        # Indices of closest observations based on DOY
        i_doy = np.argsort(
            np.mod(self.dates[self.start:self.here] -
                   self.dates[self.here + self.consecutive],
                   self.ndays))[:self.min_obs]

        _rmse = np.zeros(len(self.test_indices), np.float32)
        _X = self.X.take(i_doy, axis=0)
        for i_b, b in enumerate(self.test_indices):
            m = self.models[b]
            _rmse[i_b] = rmse(self.Y[b, :].take(i_doy), m.predict(_X))

        return _rmse
