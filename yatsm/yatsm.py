#!/usr/bin/env python
from __future__ import print_function, division

import logging
import math

import numpy as np
import pandas as pd

import statsmodels.api as sm

from glmnet.elastic_net import ElasticNet, elastic_net
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LassoLarsIC

# Some constants
ndays = 365.25

class GLMLasso(ElasticNet):

    def __init__(self, alpha=1.0):
        super(GLMLasso, self).__init__(alpha)

    def fit(self, X, y, lambdas=None):
        if lambdas is None:
            lambdas = [self.alpha]
        elif not isinstance(type(lambdas), list):
            lambdas = [lambdas]

        n_lambdas, intercept_, coef_, ia, nin, rsquared_, lambdas, _, jerr = \
            elastic_net(X, y, 1, lambdas=lambdas)
        # elastic_net will fire exception instead
        # assert jerr == 0

        self.coef_ = np.zeros(X.shape[1])
        self.coef_[ia[:nin[0]] - 1] = coef_

        self.intercept_ = intercept_
        self.rsquared_ = rsquared_

        # Create external friendly coefficients
        self.coef = np.copy(self.coef_)
        self.coef[0] += intercept_

        # Store number of observations
        self.nobs = y.size

        # Store fitted values
        self.fittedvalues = self.predict(X)

        # Calculate the residual sum of squares
        self.rss = np.sum((y - self.fittedvalues) ** 2)

        # Calculate model RMSE
        self.rmse = math.sqrt(self.rss / self.nobs)

        return self


def make_X(x, freq, intercept=True):
    """ Create X matrix of Fourier series style independent variables

    Args:
        x               base of independent variables - dates
        freq            frequency of cosine/sin waves
        intercept       include intercept in X matrix

    Output:
        X               matrix X of independent variables

    Example:
        call:
            make_X(np.array([1, 2, 3]), [1, 2])
        returns:
            array([[ 1.        ,  1.        ,  1.        ],
                   [ 1.        ,  2.        ,  3.        ],
                   [ 0.99985204,  0.99940821,  0.99866864],
                   [ 0.01720158,  0.03439806,  0.05158437],
                   [ 0.99940821,  0.99763355,  0.99467811],
                   [ 0.03439806,  0.06875541,  0.10303138]])

    """
    w = 2 * np.pi / ndays

    if intercept:
        X = np.array([np.ones_like(x), x])
    else:
        X = x

    for f in freq:
        X = np.vstack([X, np.array([
            np.cos(f * w * x),
            np.sin(f * w * x)])
        ])

    return X


def multitemp_mask(x, Y, n_year, crit=400, green=1, swir1=4):
    """ Multi-temporal cloud/shadow masking using RLM

    Returns np.array of booleans. False indicate failed mask test and should be
    masked
    """
    n_year = np.ceil(n_year)

    w = 2.0 * np.pi / ndays

    X = np.array([
        np.ones_like(x),
        np.cos(w * x),
        np.sin(w * x),
        np.cos(w / n_year * x),
        np.sin(w / n_year * x)
    ])

    green_RLM = sm.RLM(Y[green, :], X.T,
                       M=sm.robust.norms.TukeyBiweight())
    swir1_RLM = sm.RLM(Y[swir1, :], X.T,
                       M=sm.robust.norms.TukeyBiweight())

    return np.logical_and(green_RLM.fit().resid < crit,
                          swir1_RLM.fit().resid > -crit)


class YATSM(object):
    """Yet Another Time Series Model (YATSM)
    """

    def __init__(self, X, Y, consecutive=5, threshold=2.56, min_obs=None,
                 fit_indices=None, test_indices=None,
                 lassocv=False, loglevel=logging.DEBUG):
        """Initialize a YATSM model for data X (spectra) and Y (dates)

        Initialize a
        """
        # Setup logger
        logging.basicConfig()
        self.logger = logging.getLogger()
        self.logger.setLevel(loglevel)

        # Configure which implementation of LASSO we're using
        self.lassocv = lassocv
        if self.lassocv:
            self.fit_models = self.fit_models_LassoCV
            self.logger.info('Using LassoCV from sklearn')
        else:
            self.fit_models = self.fit_models_GLMnet
            self.logger.info('Using Lasso from GLMnet (lambda = 20)')

        # Store data
        self._X = X.copy()
        self._Y = Y.copy()
        self.X = X.copy()
        self.Y = Y.copy()

        # Default fitted and tested indices to all, except last band
        if fit_indices is None:
            self.fit_indices = np.arange(Y.shape[0] - 1)
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

        # Store parameters
        self.consecutive = consecutive
        self.threshold = threshold
        self.ndays = 365.25
        self.n_band = Y.shape[0]
        self.n_coef = X.shape[1]

        if min_obs is None:
            self.min_obs = int(self.n_coef * 1.5)
        else:
            self.min_obs = min_obs

        self.start = 0
        self.here = self.min_obs
        self._here = self.here

        if self.X.shape[0] < self.here + self.consecutive:
            raise Exception('Not enough observations (n = {n})'.format(
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
            ('py', 'u2')
        ])
        self.record = np.copy(self.record_template)

# POST-PROCESSING
    def merge_record(self, critF):
        """ Merge adjacent records based on nested F test """
        pass

    def omission_test(self, crit):
        """ Add omitted breakpoint into records based on residual stationarity
        """
        pass

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

        # Copy normal records
        robust_record = np.copy(self.record)
        # Update to robust model
        for i, record in enumerate(robust_record):
            # Find matching X and Y in data
            index = np.where((self.X[:, 1] >= record['start']) &
                             (self.X[:, 1] <= record['end']))[0]
            # Grab matching X and Y
            _X = self.X[index, :]
            _Y = self.Y[:, index]

            # Refit each band
            for i_b, b in enumerate(self.fit_indices):
                # Find nonzero
                nonzero = np.where(robust_record[i]['coef'][:, i_b] != 0)[0]

                if nonzero.size == 0:
                    continue

                # Setup model
                rirls_model = sm.RLM(_Y[b, :], _X[:, nonzero],
                                     M=sm.robust.norms.TukeyBiweight())
                # Fit
                fit = rirls_model.fit()
                # Store updated coefficients
                robust_record[i]['coef'][nonzero, i_b] = fit.params

                # Update RMSE
                rss = np.sum((fit.resid) ** 2)
                robust_record[i]['rmse'][i_b] = math.sqrt(rss / index.size)

            self.logger.debug('Updated record {i} to robust results'.
                              format(i=i))

        return robust_record

    def reset(self):
        """ Resets 'start' and 'here' indices """
        self.start = 0
        self.here = self.min_obs
        self._here = self.here
        self.ran = False

    @property
    def span_time(self):
        """ Return time span (in days) between start and end of model """
        return abs(self.X[self.here, 1] - self.X[self.start, 1])

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
        return self.here <= self.X.shape[0] - self.consecutive - 1

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

        self.ran = True
        # Deal with end of time series

    def train(self):
        """ Train time series model """
        # Test if we can train yet
        if self.span_time <= self.ndays or self.span_index < self.n_coef:
            self.log_debug('could not train - moving forward')
            return

        # Multitemporal noise removal
        mask = np.ones(self.X.shape[0], dtype=np.uint8)
        index = np.arange(self.start, self.here + self.consecutive + 1,
                          dtype=np.uint16)
        mask[index] = multitemp_mask(self.X[index, 1], self.Y[:, index],
                                     self.span_time)

        self.log_debug('Multitemporal masking - {i} / {n} masked'.format(
            i=(mask[index] == 0).sum(),
            n=mask[index].shape[0]))

        # Check if there are enough observations for model with noise removed
        span_index = mask[index][:-self.consecutive].sum().astype(np.uint8)

        span_time = abs(self.X[mask == 1, 1][index[-self.consecutive]] -
                        self.X[mask == 1, 1][index[0]])

        self.log_debug('span_index: {si}, span_time: {st}'.format(
            si=span_index, st=span_time))

        if span_index < self.min_obs:
            self.log_debug('Multitemporal masking - not enough obs ({n})'.
                           format(n=span_index))
            return
        if span_time < self.ndays:
            self.log_debug('Multitemporal masking - not enough time ({t})'.
                           format(t=span_time))
            return

        # There is enough time in train period to fit - remove noise
        self._X = self.X[mask == 1, :]
        self._Y = self.Y[:, mask == 1]

        self.log_debug('Removed multitemporal noise ({b} to {n})'.format(
            b=self.X.shape[0], n=self._X.shape[0]))

        # record our current position
        #   important for next iteration of noise removal
        self._here = self.here

        self.log_debug('span index {si}'.format(si=span_index))
        self.here = self.start + span_index - 1
        self.log_debug('Updated "here"')

        # After noise removal, try to fit models
        models = self.fit_models(self._X, self._Y, bands=self.test_indices)

        start_resid = np.zeros(len(self.test_indices))
        end_resid = np.zeros(len(self.test_indices))
        for i, (b, m) in enumerate(zip(self.test_indices, models)):
            start_resid[i] = np.abs(self._Y[b, self.start] -
                                    m.predict(self._X[self.start, :])) / m.rmse
            end_resid[i] = np.abs(self._Y[b, self.here] -
                                  m.predict(self._X[self.here, :])) / m.rmse

        if np.linalg.norm(start_resid) > self.threshold or \
                np.linalg.norm(end_resid) > self.threshold:
            self.log_debug('Training period unstable')

            self.start += 1
            self.here = self._here
            return

        self.X = self._X
        self.Y = self._Y

        self.log_debug('Entering monitoring period')

        self.monitoring = True

    def update_model(self):
        # Only train once a year
        if abs(self.X[self.here, 1] - self.trained_date) > self.ndays:
            self.log_debug('Monitoring - retraining ({n} days since last)'.
                           format(n=self.X[self.here, 1] - self.trained_date))

            # Fit timeseries models
            self.models = self.fit_models(self.X, self.Y)

            # Update record
            self.record[self.n_record]['start'] = self.X[self.start, 1]
            self.record[self.n_record]['end'] = self.X[self.here, 1]
            for i, m in enumerate(self.models):
#                Pdb().set_trace()
                self.record[self.n_record]['coef'][:, i] = m.coef
                self.record[self.n_record]['rmse'][i] = m.rmse
            self.log_debug('Monitoring - updated ')

            self.trained_date = self.X[self.here, 1]
        else:
            # Update record with new end date
            self.record[self.n_record]['end'] = self.X[self.here, 1]

    def monitor(self):
        """ Monitor for changes in time series """
        # Store test scores
        scores = np.zeros((self.consecutive, len(self.test_indices)),
                          dtype=np.float32)

        for i in range(self.consecutive):
            for i_b, b in enumerate(self.test_indices):
                m = self.models[b]
                # Get test score for future observations
                scores[i, i_b] = (np.abs(self.Y[b, self.here + i] -
                                         m.predict(self.X[self.here + i, :])) /
                                  m.rmse)

        # Check for scores above critical value
        mag = np.linalg.norm(scores, axis=1)

        if np.all(mag > self.threshold):
            self.log_debug('CHANGE DETECTED')

            self.record[self.n_record]['break'] = self.X[self.here + 1, 1]

            self.record = np.append(self.record, self.record_template)
            self.n_record += 1
            self.start = self.here + 1

            self.monitoring = False

    def fit_models_GLMnet(self, X, Y, index=None, bands=None):
        """ Try to fit models to training period time series """
        if bands is None:
            bands = self.fit_indices

        if index is None:
            index = np.arange(self.start, self.here + 1)

        models = []

        for b in bands:
            lasso = GLMLasso()
            lasso = lasso.fit(X[index, :], Y[b, index], lambdas=20)

            models.append(lasso)

        return np.array(models)

    def fit_models_LassoCV(self, X, Y, index=None, bands=None):
        """ Try to fit models to training period time series """
        if bands is None:
            bands = self.fit_indices

        if index is None:
            index = np.arange(self.start, self.here + 1)

        models = []

        for b in bands:
            # lasso = LassoCV(n_alphas=100)
            # lasso = LassoLarsCV(masx_n_alphas=100)
            lasso = LassoLarsIC(criterion='bic')
            lasso = lasso.fit(X[index, :], Y[b, index])
            lasso.nobs = Y[b, index].size
            lasso.coef = np.copy(lasso.coef_)
            lasso.coef[0] += lasso.intercept_
            lasso.fittedvalues = lasso.predict(X[index, :])
            lasso.rss = np.sum((Y[b, index] - lasso.fittedvalues) ** 2)
            lasso.rmse = math.sqrt(lasso.rss / lasso.nobs)
#            lasso = GLMLasso()
#            lasso = lasso.fit(X[index, :], Y[b, index], lambdas=20)

            models.append(lasso)

        return np.array(models)

    def train_plot_debug(self, mask, index):
        """ Training / historical period multitemporal cloud masking debug """
        from ggplot import ggplot, geom_point, xlab, ylab, ggtitle, aes

        cols = np.repeat('clear', index.shape[0])
        cols[mask[index] == 0] = 'noise'
        df = pd.DataFrame({'X': self.X[index, 1],
                           'Y': self.Y[4, index],
                           'mask': cols
                           })
        print(ggplot(aes('X', 'Y', color='mask'), df) +
              geom_point() +
              xlab('Ordinal Date') +
              ylab('B5 Reflectance') +
              ggtitle('Cloud Screening - segment: {i}'.format(i=self.n_record))
              )

    def monitor_plot_debug(self, index, model, i_buffer=10):
        """ Monitoring debug plot """
        import matplotlib.pyplot as plt
        # Show before/after current timeseries
        before_buffer = max(0, index[0] - i_buffer)
        after_buffer = min(self.X[:, 1].size - 1, index[-1] + i_buffer)

        plt.plot(self.X[before_buffer:after_buffer, 1],
                 self.Y[4, before_buffer:after_buffer], 'ko')

        pred_x = np.arange(self.X[before_buffer, 1],
                           self.X[after_buffer, 1])
        pred_X = make_X(pred_x).T
        plt.plot(pred_x, model.predict(pred_X), '--', color='0.75')

        # Show monitoring prediction
        pred_x = np.arange(self.X[index[0], 1], self.X[index[-1], 1])
        pred_X = make_X(pred_x).T
        plt.plot(pred_x, model.predict(pred_X))

        # Show currently considered obs
        plt.plot(self.X[index, 1], self.Y[4, index], 'ro')

        plt.title('Model {i} - RMSE: {rmse}'.format(i=self.n_record,
                                                    rmse=round(model.rmse, 3)))

        plt.show()

    def log_debug(self, message):
        """ Custom logging message """
        self.logger.debug('{start},{here} ({si},{st}) : ({trained}) : '.format(
            start=self.start, here=self.here,
            si=self.span_index, st=self.span_time,
            trained=self.monitoring) +
            message)
