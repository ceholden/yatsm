from __future__ import print_function, division

import logging
import math
import sys

import numpy as np
import numpy.lib.recfunctions

import scipy.linalg
import scipy.stats
from sklearn.linear_model import LassoLarsIC  # , Lasso, LassoCV, LassoLarsCV
import statsmodels.api as sm

from errors import TSLengthException
from masking import smooth_mask, multitemp_mask
from regression.glmnet_fit import GLMLasso
from regression import robust_fit as rlm
from utils import date2index


class YATSM(object):
    """Initialize a YATSM model for data X (spectra) and Y (dates)

    YATSM model based off of tests for structural changes from the
    econometrics literature including the MOSUM or CUMSUM (Chu et al,
    Zeileis, and others) as implemented in a remote sensing context by
    BFAST (Verbesselt, et al. 2012) and CCDC (Zhu and Woodcock, 2014). This
    effort is not intended as a direct port of either algorithms.

    Args:
      X (ndarray): Independent variable matrix
      Y (ndarray): Dependent variable matrix
      consecutive (int): Consecutive observations to trigger change
      threshold (float): Test statistic threshold for change
      min_obs (int): Minimum observations in model
      min_rmse (float): Minimum RMSE for models during testing
      fit_indices (ndarray): Indices of Y to fit models for
      test_indices (ndarray): Indices of Y to test for change with
      retrain_time (float): Number of days between model fit updates during
        monitoring period
      screening (str): Style of prescreening of the timeseries
        for noise. Options are 'RLM' or 'LOWESS'
      screening_crit (float, optional): critical value for multitemporal
        noise screening
      green_band (int, optional): Index of green band in Y for
        multitemporal masking (default: 1)
      swir1_band (int, optional): Index of first SWIR band in Y for
        multitemporal masking (default: 4)
      remove_noise (bool, optional): Remove observation if change is not
        detected but first observation is above threshold (if it looks like
        noise) (default: True)
      dynamic_rmse (bool, optional): Vary RMSE as a function of day of year (
        default: False)
      lassocv (bool, optional): Use scikit-learn LarsLassoCV over glmnet
      design_info (patsy.DesignInfo, optional): design information for X, if
        X is created using Patsy
      px (int, optional): X (column) pixel reference
      py (int, optional): Y (row) pixel reference
      logger (logging.Logger, optional): Specific logger to use, else get one

    """

    ndays = 365.25
    green_band = 1
    swir1_band = 4
    screening_types = ['RLM', 'LOWESS']

    def __init__(self, X, Y,
                 consecutive=5, threshold=2.56, min_obs=None, min_rmse=None,
                 fit_indices=None, test_indices=None, retrain_time=ndays,
                 screening='RLM', screening_crit=400.0,
                 green_band=green_band, swir1_band=swir1_band,
                 remove_noise=True, dynamic_rmse=False,
                 lassocv=False,
                 design_info=None, px=0, py=0,
                 logger=None):
        # Setup logger
        self.logger = logger or logging.getLogger('yatsm')

        # Configure which implementation of LASSO we're using
        self.lassocv = lassocv
        if self.lassocv:
            self.fit_models = self.fit_models_LassoCV
            self.logger.info('Using LassoCV from sklearn')
        else:
            self.fit_models = self.fit_models_GLMnet
            self.logger.info('Using Lasso from GLMnet (lambda = 20)')

        # Store data
        self.X = X
        self.Y = Y

        # Find column index of X containing date from Patsy
        if design_info:
            self.design_info = design_info
            if 'x' not in design_info.term_name_slices.keys():
                raise AttributeError('Design info must specify "x" (slope)')
            self._jx = design_info.term_name_slices['x'].start
        else:
            self._jx = 1

        # Default fitted and tested indices to all, except last band
        if fit_indices is None:
            self.fit_indices = np.arange(Y.shape[0])
        else:
            if max(fit_indices) < Y.shape[0]:
                self.fit_indices = fit_indices
            else:
                raise IndexError('Specified fit_indices larger than Y matrix')

        if test_indices is None:
            self.test_indices = np.arange(Y.shape[0])
        else:
            if max(test_indices) < Y.shape[0]:
                self.test_indices = test_indices
            else:
                raise IndexError('Specified test_indices larger than Y matrix')

        self.retrain_time = retrain_time

        # Type of noise screening
        if screening not in self.screening_types:
            raise TypeError('Unknown screening type')
        # Define method according to type
        if screening == 'RLM':
            self.screen_timeseries = self.screen_timeseries_RLM
            self.logger.debug('Using RLM for screening')
        elif screening == 'LOWESS':
            self.screen_timeseries = self.screen_timeseries_LOWESS
            self.logger.debug('Using LOWESS for screening')
        # Keep track if timeseries has been screened for full-TS LOWESS
        self.screened = False

        self.green_band = green_band
        self.swir1_band = swir1_band

        self.screening_crit = screening_crit

        self.remove_noise = remove_noise

        if dynamic_rmse:
            self.get_rmse = self.get_dynamic_rmse
        else:
            self.get_rmse = self.get_model_rmse

        # Attributes
        self.n_band = Y.shape[0]
        self.n_coef = X.shape[1]

        # Store parameters
        self.consecutive = consecutive
        self.threshold = threshold

        if min_obs is None:
            self.min_obs = int(self.n_coef * 1.5)
        else:
            self.min_obs = min_obs

        # Minimum RMSE to prevent being overly sensitive to changes
        if min_rmse:
            self.min_rmse = min_rmse
        else:
            # if None, set to max float size so it never is minimum
            self.min_rmse = sys.float_info.min

        # Index of time segment location
        self.start = 0
        self.here = self.min_obs
        self._here = self.here

        if self.X.shape[0] < self.here + self.consecutive:
            raise TSLengthException('Not enough observations (n = {n})'.format(
                n=self.X.shape[0]))

        # Record if model has been trained
        self.monitoring = False
        # Record if model has been ran
        self.ran = False

        # Store array of time series model (GLMnet or LassoCV)
        self.models = []

        self.n_record = 0
        self.record_template = np.zeros(1, dtype=[
            ('start', 'i4'),
            ('end', 'i4'),
            ('break', 'i4'),
            ('coef', 'float32', (self.n_coef, len(self.fit_indices))),
            ('rmse', 'float32', len(self.fit_indices)),
            ('px', 'u2'),
            ('py', 'u2'),
            ('magnitude', 'float32', len(self.fit_indices))
        ])
        self.record_template['px'][0] = px
        self.record_template['py'][0] = py
        self.record = np.copy(self.record_template)

# POST-PROCESSING
    def commission_test(self, alpha=0.001):
        """ Merge adjacent records based on Chow Tests for nested models

        Use Chow Test to find false positive, spurious, or unnecessary breaks
        in the timeseries by comparing the effectiveness of two separate
        adjacent models with one single model that spans the entire time
        period.

        Chow test is described:

        .. math::
            \\frac{[RSS_r - (RSS_1 + RSS_2)] / k}{(RSS_1 + RSS_2) / (n - 2k)}

        where
            - :math:`RSS_u` is the RSS of the combined, or, restricted model
            - :math:`RSS_1` is the RSS of the first model
            - :math:`RSS_2` is the RSS of the second model
            - :math:`k` is the number of model parameters
            - :math:`n` is the number of total observations

        Because we look for change in multiple bands, the RSS used to compare
        the unrestricted versus restricted models is the L2 norm of RSS
        values from `self.test_indices`.

        Args:
          alpha (float): significance level for F-statistic (default: 0.01)

        Returns:
          np.ndarray: updated copy of `self.models` with spurious models
            combined into unified model

        """
        if self.record.size == 1:
            return self.record

        k = self.n_coef

        models = []
        merged = False

        for i in range(len(self.record) - 1):
            if merged:
                m_1 = models[-1]
            else:
                m_1 = self.record[i]
            m_2 = self.record[i + 1]

            m_1_start = date2index(self.X[:, self._jx], m_1['start'])
            m_1_end = date2index(self.X[:, self._jx], m_1['end'])
            m_2_start = date2index(self.X[:, self._jx], m_2['start'])
            m_2_end = date2index(self.X[:, self._jx], m_2['end'])

            m_r_start = m_1_start
            m_r_end = m_2_end

            n = m_r_end - m_r_start

            F_crit = scipy.stats.f.ppf(1 - alpha, k, n - 2 * k)

            m_1_rss = np.zeros(self.test_indices.size)
            m_2_rss = np.zeros(self.test_indices.size)
            m_r_rss = np.zeros(self.test_indices.size)

            for i_b, b in enumerate(self.test_indices):
                m_1_rss[i_b] = scipy.linalg.lstsq(
                    self.X[m_1_start:m_1_end, :],
                    self.Y[b, m_1_start:m_1_end])[1]
                m_2_rss[i_b] = scipy.linalg.lstsq(
                    self.X[m_2_start:m_2_end, :],
                    self.Y[b, m_2_start:m_2_end])[1]
                m_r_rss[i_b] = scipy.linalg.lstsq(
                    self.X[m_r_start:m_r_end, :],
                    self.Y[b, m_r_start:m_r_end])[1]

            m_1_rss = np.linalg.norm(m_1_rss)
            m_2_rss = np.linalg.norm(m_2_rss)
            m_r_rss = np.linalg.norm(m_r_rss)

            F = ((m_r_rss - (m_1_rss + m_2_rss)) / k) / \
                ((m_1_rss + m_2_rss) / (n - 2 * k))

            if F > F_crit:
                # Reject H0 and retain change
                # Only add in previous model if first model
                if i == 0:
                    models.append(m_1)
                models.append(m_2)
                merged = False
            else:
                # Fail to reject H0 -- ignore change and merge
                m_new = np.copy(self.record_template)[0]

                # Remove last previously added model from list to merge
                if i != 0:
                    del models[-1]

                m_new['start'] = m_1['start']
                m_new['end'] = m_2['end']
                m_new['break'] = m_2['break']

                _models = self.fit_models(
                    self.X[m_r_start:m_r_end, :],
                    self.Y[:, m_r_start:m_r_end])

                for i_m, _m in enumerate(_models):
                    m_new['coef'][:, i_m] = _m.coef
                    m_new['rmse'][i_m] = _m.rmse

                # Preserve magnitude from 2nd model that was merged
                m_new['magnitude'] = m_2['magnitude']

                models.append(m_new)

                merged = True

        return np.array(models)

    def omission_test(self, crit=0.05, behavior='ANY',
                      indices=None):
        """ Add omitted breakpoint into records based on residual stationarity

        Uses recursive residuals within a CUMSUM test to check if each model
        has omitted a "structural change" (e.g., land cover change). Returns
        an array of True or False for each timeseries segment record depending
        on result from `statsmodels.stats.diagnostic.breaks_cusumolsresid`.

        Args:
          crit (float, optional): Critical p-value for rejection of null
            hypothesis that data contain no structural change
          behavior (str, optional): Method for dealing with multiple
            `test_indices`. `ANY` will return True if any one test index
            rejects the null hypothesis. `ALL` will only return True if ALL
            test indices reject the null hypothesis.
          indices (np.ndarray, optional): Array indices to test. User provided
            indices must be a subset of `self.test_indices`.

        Returns:
          np.ndarray: Array of True or False for each record where
            True indicates omitted break point

        """
        if behavior.lower() not in ['any', 'all']:
            raise ValueError('`behavior` must be "any" or "all"')

        if not indices:
            indices = self.test_indices

        if not np.all(np.in1d(indices, self.test_indices)):
            raise ValueError('`indices` must be a subset of '
                             '`self.test_indices`')

        if not self.ran:
            return np.empty(0, dtype=bool)

        omission = np.zeros((self.record.size, len(indices)),
                            dtype=bool)

        for i, r in enumerate(self.record):
            # Skip if no model fit
            if r['start'] == 0 or r['end'] == 0:
                continue
            # Find matching X and Y in data
            index = np.where(
                (self.X[:, self._jx] >= min(r['start'], r['end'])) &
                (self.X[:, self._jx] <= max(r['end'], r['start'])))[0]
            # Grab matching X and Y
            _X = self.X[index, :]
            _Y = self.Y[:, index]

            for i_b, b in enumerate(indices):
                # Create OLS regression
                ols = sm.OLS(_Y[b, :], _X).fit()
                # Perform CUMSUM test on residuals
                test = sm.stats.diagnostic.breaks_cusumolsresid(
                    ols.resid, _X.shape[1])

                if test[1] < crit:
                    omission[i, i_b] = True
                else:
                    omission[i, i_b] = False

        # Collapse band answers according to `behavior`
        if behavior.lower() == 'any':
            return np.any(omission, 1)
        else:
            return np.all(omission, 1)

    @property
    def robust_record(self):
        """ Returns a copy of YATSM record output with robustly fitted models

        After YATSM has been run, take each time segment and re-fit the model
        using robust iteratively reweighted least squares (RIRLS) regression.
        RIRLS will only be performed using non-zero coefficients from original
        regression.

        The returned model results should be more representative of the
        signal found because it will remove influence of outlying observations,
        such as clouds or shadows.

        If YATSM has not yet been run, returns None
        """
        if not self.ran:
            return None

        # Create new array for robust coefficients and RMSE
        robust = np.zeros(self.record.shape[0], dtype=[
            ('robust_coef', 'float32', (self.n_coef, len(self.fit_indices))),
            ('robust_rmse', 'float32', len(self.fit_indices)),
        ])

        # Update to robust model
        for i, r in enumerate(self.record):
            # Find matching X and Y in data
            index = np.where(
                (self.X[:, self._jx] >= min(r['start'], r['end'])) &
                (self.X[:, self._jx] <= max(r['end'], r['start'])))[0]
            # Grab matching X and Y
            _X = self.X[index, :]
            _Y = self.Y[:, index]

            # Refit each band
            for i_b, b in enumerate(self.fit_indices):
                # Find nonzero
                nonzero = np.nonzero(self.record[i]['coef'][:, i_b])[0]

                if nonzero.size == 0:
                    continue

                # Setup model
                rirls_model = rlm.RLM(_Y[b, :], _X[:, nonzero],
                                      M=rlm.bisquare)

                # Fit
                fit = rirls_model.fit()
                # Store updated coefficients
                robust[i]['robust_coef'][nonzero, i_b] = fit.coefs

                # Update RMSE
                robust[i]['robust_rmse'][i_b] = \
                    math.sqrt(rirls_model.rss / index.size)

            self.logger.debug('Updated record {i} to robust results'.
                              format(i=i))

        # Merge
        robust_record = np.lib.recfunctions.merge_arrays((self.record, robust),
                                                         flatten=True)

        return robust_record

    def reset(self):
        """ Resets 'start' and 'here' indices """
        self.n_record = 0
        self.record = np.copy(self.record_template)
        self.start = 0
        self.here = self.min_obs
        self._here = self.here
        self.ran = False

# HELPER PROPERTIES
    @property
    def span_time(self):
        """ Return time span (in days) between start and end of model """
        return abs(self.X[self.here, self._jx] -
                   self.X[self.start, self._jx])

    @property
    def span_index(self):
        """ Return time span (in index) between start and end of model """
        return (self.here - self.start)

    @property
    def running(self):
        """ Determine if timeseries can run """
        return self.here < self.X.shape[0]

    @property
    def can_monitor(self):
        """ Determine if timeseries can monitor the future consecutive obs """
        return self.here < self.X.shape[0] - self.consecutive - 1

# MAIN LOOP
    def run(self):
        """ Run timeseries model """
        # Record date of last time model was trained
        self.trained_date = 0

        while self.running:

            while not self.monitoring and self.can_monitor:
                self.train()
                self.here += 1

            while self.monitoring and self.can_monitor:
                # Update model if required
                self.update_model()
                # Perform monitoring check
                self.monitor()
                # Iterate forward
                self.here += 1

            self.here += 1

        # If we ended without being able to monitor again, delete last model
        # since it will be empty
        if self.record[-1]['start'] == 0 and self.record[-1]['end'] == 0:
            self.record = self.record[:-1]

        self.ran = True

        # Deal with start and end of time series #TODO

    def screen_timeseries_LOWESS(self, span=None):
        """ Screen entire dataset for noise before training using LOWESS

        Args:
          span (int, optional): span for LOWESS

        Returns:
          bool: True if timeseries is screened and we can train, else False

        """
        if not self.screened:
            if not span:
                span = self.consecutive * 2 + 1

            mask = smooth_mask(self.X[:, self._jx], self.Y, span,
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

    def screen_timeseries_RLM(self):
        """ Screen training period for noise with IRWLS RLM

        Returns:
          bool: True if timeseries is screened and we can train, else False

        """
        # Multitemporal noise removal
        mask = np.ones(self.X.shape[0], dtype=np.bool)
        index = np.arange(self.start, self.here + self.consecutive,
                          dtype=np.uint16)
        mask[index] = multitemp_mask(self.X[index, self._jx],
                                     self.Y[:, index],
                                     self.span_time / self.ndays,
                                     crit=self.screening_crit,
                                     green=self.green_band,
                                     swir1=self.swir1_band)

        # Check if there are enough observations for model with noise removed
        _span_index = mask[index][:-self.consecutive].sum()

        # Return if not enough observations
        if _span_index < self.min_obs:
            self.logger.debug('    multitemp masking - not enough obs')
            return False

        # There is enough observations in train period to fit - remove noise
        self._X = self.X[mask, :]
        self._Y = self.Y[:, mask]

        # record our current position
        #   important for next iteration of noise removal
        self._here = self.here

        # Go forward after noise removal
        self.here = self.start + _span_index - 1

        if self.span_time < self.ndays:
            self.logger.debug('    multitemp masking - not enough time')
            self.here = self._here
            return False

        self.logger.debug('Updated "here"')

        return True

    def train(self):
        """ Train time series model """
        # Test if we can train yet
        if self.span_time <= self.ndays or self.span_index < self.n_coef:
            self.logger.debug('could not train - moving forward')
            return

        # Check if screening was OK
        if not self.screen_timeseries():
            return

        # Test if we can still run after noise removal
        if self.here >= self._X.shape[0]:
            self.logger.debug(
                'Not enough observations to proceed after noise removal')
            raise TSLengthException(
                'Not enough observations after noise removal')

        # After noise removal, try to fit models
        models = self.fit_models(self._X[self.start:self.here + 1, :],
                                 self._Y[:, self.start:self.here + 1],
                                 bands=self.test_indices)

        # Ensure first and last points aren't unusual
        start_resid = np.zeros(len(self.test_indices))
        end_resid = np.zeros(len(self.test_indices))
        for i, (b, m) in enumerate(zip(self.test_indices, models)):
            start_resid[i] = (np.abs(self._Y[b, self.start] -
                                     m.predict(self._X[self.start, :])) /
                              max(self.min_rmse, m.rmse))
            end_resid[i] = (np.abs(self._Y[b, self.here] -
                                   m.predict(self._X[self.here, :])) /
                            max(self.min_rmse, m.rmse))

        if np.linalg.norm(start_resid) > self.threshold or \
                np.linalg.norm(end_resid) > self.threshold:
            self.logger.debug('Training period unstable')
            self.start += 1
            self.here = self._here
            return

        self.X = self._X
        self.Y = self._Y

        self.logger.debug('Entering monitoring period')

        self.monitoring = True

    def update_model(self):
        # Only train if enough time has past
        if abs(self.X[self.here, self._jx] - self.trained_date) > \
                self.retrain_time:
            self.logger.debug('Monitoring - retraining ({n} days since last)'.
                              format(n=self.X[self.here, self._jx] -
                                     self.trained_date))

            # Fit timeseries models
            self.models = self.fit_models(self.X[self.start:self.here + 1, :],
                                          self.Y[:, self.start:self.here + 1])

            # Update record
            self.record[self.n_record]['start'] = self.X[self.start, self._jx]
            self.record[self.n_record]['end'] = self.X[self.here, self._jx]
            for i, m in enumerate(self.models):
                self.record[self.n_record]['coef'][:, i] = m.coef
                self.record[self.n_record]['rmse'][i] = m.rmse
            self.logger.debug('Monitoring - updated ')

            self.trained_date = self.X[self.here, self._jx]
        else:
            # Update record with new end date
            self.record[self.n_record]['end'] = self.X[self.here, self._jx]

    def monitor(self):
        """ Monitor for changes in time series """
        # Store test scores
        scores = np.zeros((self.consecutive, len(self.test_indices)),
                          dtype=np.float32)

        rmse = self.get_rmse()

        for i in range(self.consecutive):
            for i_b, b in enumerate(self.test_indices):
                m = self.models[b]
                # Get test score for future observations
                scores[i, i_b] = (
                    (self.Y[b, self.here + i] -
                        m.predict(self.X[self.here + i, :])) /
                    max(self.min_rmse, rmse[i_b])
                )

        # Check for scores above critical value
        mag = np.linalg.norm(np.abs(scores), axis=1)

        if np.all(mag > self.threshold):
            self.logger.debug('CHANGE DETECTED')

            # Record break date
            self.record[self.n_record]['break'] = \
                self.X[self.here + 1, self._jx]
            # Record magnitude of difference for tested indices
            self.record[self.n_record]['magnitude'][self.test_indices] = \
                np.mean(scores, axis=0)

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
            self.here -= 1

    def fit_models_GLMnet(self, X, Y, bands=None):
        """ Try to fit models to training period time series """
        if bands is None:
            bands = self.fit_indices

        models = []

        for b in bands:
            lasso = GLMLasso()
            lasso = lasso.fit(X, Y[b, :], lambdas=20)

            models.append(lasso)

        return np.array(models)

    def fit_models_LassoCV(self, X, Y, bands=None):
        """ Try to fit models to training period time series """
        if bands is None:
            bands = self.fit_indices

        models = []

        for b in bands:
            # lasso = LassoCV(n_alphas=100)
            # lasso = LassoLarsCV(masx_n_alphas=100)
            lasso = LassoLarsIC(criterion='bic')
            lasso = lasso.fit(X, Y[b, :])
            lasso.nobs = Y[b, :].size
            lasso.coef = np.copy(lasso.coef_)
            lasso.coef[0] += lasso.intercept_
            lasso.fittedvalues = lasso.predict(X)
            lasso.rss = np.sum((Y[b, :] - lasso.fittedvalues) ** 2)
            lasso.rmse = math.sqrt(lasso.rss / lasso.nobs)

            models.append(lasso)

        return np.array(models)

    def get_model_rmse(self):
        """ Return the normal RMSE of each fitted model

        Returns:
          np.ndarray: NumPy array containing RMSE of each tested model

        """
        return np.array([m.rmse for m in self.models])[self.test_indices]

    def get_dynamic_rmse(self):
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
        i_doy = np.argsort(np.mod(
            self.X[self.start:self.here, self._jx] -
            self.X[self.here + self.consecutive, self._jx],
            self.ndays))[:self.min_obs]

        rmse = np.zeros(len(self.test_indices), np.float32)
        for i_b, b in enumerate(self.test_indices):
            m = self.models[b]
            rmse[i_b] = np.sqrt(np.sum(
                (self.Y[b, :].take(i_doy) -
                    m.predict(self.X.take(i_doy, axis=0))) ** 2)
                / i_doy.size)

        return rmse
