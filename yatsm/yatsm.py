#!/usr/bin/env python
""" Yet Another Time Series Model

Usage:
    yatsm.py [options] <location> <px> <py>

Options:
    --consecutive=<n>       Consecutive observations to find change [default: 5]
    --threshold=<T>         Threshold for change [default: 2.56]
    --min_obs=<n>           Min number of obs per model [default: 1.5 * n_coef]
    --freq=<freq>           Sin/cosine frequencies [default: 1 2 3]
    --lassocv               Use sklearn cross-validated LassoCV
    --reverse               Run timeseries in reverse
    --plot_band=<b>         Band to plot for diagnostics [default: 5]
    -h --help               Show help

Example:


"""
from __future__ import print_function, division

from datetime import datetime as dt
import logging
import math

from docopt import docopt

import numpy as np
import pandas as pd

import statsmodels.api as sm

from glmnet.elastic_net import ElasticNet, elastic_net
from sklearn.linear_model import Lasso, LassoCV

from ggplot import *

from ts_driver.timeseries_ccdc import py2mldate, ml2pydate

from IPython.core.debugger import Pdb

import matplotlib.pyplot as plt

# Some constants
ndays = 365.25
fmask = 7

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
        self.coef[0] = intercept_

        # Store number of observations
        self.nobs = y.size

        # Store fitted values
        self.fittedvalues = self.predict(X)

        # Calculate the residual sum of squares
        self.rss = np.sum((y - self.fittedvalues) ** 2)

        # Calculate model RMSE
        self.rmse = math.sqrt(self.rss / self.nobs)

        return self

def make_X(x):
    w = 2 * np.pi / ndays

    return np.array([
        np.ones_like(x),
        x,
        np.cos(w * x),
        np.sin(w * x),
        np.cos(2 * w * x),
        np.sin(2 * w * x),
        np.cos(3 * w * x),
        np.sin(3 * w * x)
    ])


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

    """ docstring """

    def __init__(self, X, Y, consecutive=5, threshold=2.56, min_obs=None):
        """
        :param df: Pandas dataframe of model and observations
        :param consecutive: consecutive observations for change
        :threshold: "t-statistic-like" threshold for magnitude of change
        :min_obs: minimum number of observations for time series initialization
        """
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger()

        # Store data
        self._X = X
        self._Y = Y
        self.X = X
        self.Y = Y
        # Store parameters
        self.consecutive = consecutive
        self.threshold = threshold

        self.fit_indices = np.array([0, 1, 2, 3, 4, 5, 6])
        self.test_indices = np.array([2, 3, 4])

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

        # Store array of time series model (GLMnet or LassoCV)
        self.models = []

        self.n_record = 0
        self.record_template = np.zeros(1, dtype=[
            ('start', 'u4'),
            ('end', 'u4'),
            ('break', 'u4'),
            ('coef', 'float32', (self.n_coef, len(self.fit_indices))),
            ('rmse', 'float32', len(self.fit_indices)),
            ('px', 'u2'),
            ('py', 'u2')
        ])
        self.record = np.copy(self.record_template)

    @property
    def span_time(self):
        """ Return time span (in days) between start and end of model """
        return (self.X[self.here, 1] - self.X[self.start, 1])


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

        while self.can_monitor:

            while not self.monitoring:
                self.train()
                self.here += 1
                self.log_debug('Updated here')

            while self.monitoring and self.can_monitor:
                # Update model if required
                self.update_model()
                # Perform monitoring check
                self.monitor()
                # Iterate forward
                self.here += 1
                self.log_debug('Updated here')

#            if self.here >= self.X[:, 1].size - self.consecutive - 1:


    def train_plot_debug(self, mask, index):
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
              ggtitle('Cloud Screening - segment: {i}'.format(i=self.n_record)))


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

        span_time = (self.X[mask == 1, 1][index[-self.consecutive]] -
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

        # self.train_plot_debug(mask, index)

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

#        df = pd.DataFrame({
#            'X': self._X[self.start:self.here + 1, 1],
#            'Y': self._Y[4, self.start:self.here + 1],
#            'pred': m.predict(self._X[self.start:self.here + 1, :])
#        })
#        print(ggplot(aes('X', 'Y'), df) + geom_point() +
#              geom_line(aes('X', 'pred'), df, color='red'))

        self.X = self._X
        self.Y = self._Y

        self.log_debug('Entering monitoring period')

        self.monitoring = True

    def update_model(self):
        # Only train once a year
        if self.X[self.here, 1] - self.trained_date > self.ndays:
            print('DEBUG TRAINING\%\%\%\%\%\%\%\%')

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


    def monitor_plot_debug(self, index, model, i_buffer=10):
        """ Monitoring debug plot """
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


    def monitor(self):
        """ Monitor for changes in time series """
        # Current window
        index = np.arange(self.start, self.here + 1)
        # Prediction window
        test_index = np.arange(self.here, self.here + self.consecutive)

        # Store test scores
        scores = np.zeros((self.consecutive, len(self.test_indices)),
                          dtype=np.float32)

        for i in range(self.consecutive):
            for i_b, b in enumerate(self.test_indices):
                m = self.models[b]
                # Get test score for future observations
                scores[i, i_b] = np.abs(self.Y[b, self.here + i] -
                    m.predict(self.X[self.here + i, :])) / m.rmse

        # Check for scores above critical value
        mag = np.linalg.norm(scores, axis=1)

        # self.monitor_plot_debug(index, m)

        if np.all(mag > self.threshold):
            self.log_debug('CHANGE DETECTED')

            self.record[self.n_record]['break'] = self.X[self.here + 1, 1]

            self.record = np.append(self.record, self.record_template)
            self.n_record += 1
            self.start = self.here + 1

            self.monitoring = False

    def fit_models(self, X, Y, index=None, bands=None):
        """ Try to fit models to training period time series """
        if bands is None:
            bands = self.fit_indices

        if index is None:
            index = np.arange(self.start, self.here + 1)

        models = []

        for b in bands:
#            lasso = LassoCV(n_alphas=50)
            lasso = GLMLasso()
            lasso = lasso.fit(X[index, :], Y[b, index], lambdas=20)

            models.append(lasso)

        return np.array(models)

    def log_debug(self, message):
        """ Custom logging message """
        self.logger.debug('{start},{here} ({si},{st}) : ({trained}) : '.format(
            start=self.start, here=self.here,
            si=self.span_index, st=self.span_time,
            trained=self.monitoring) +
            message)


if __name__ == '__main__':
    x = np.load('sample/sample_x.npy')
    Y = np.load('sample/px96_py91_Y.npy')
    # Y = np.load('sample/px61_py75_Y.npy')

    # Filter out time series
    # Remove Fmask
    clear = Y[7, :] <= 1
    Y = Y[:, clear][:fmask, :]
    x = x[clear]

    # Simulate some clouds
    Y[4, 0] = Y[4, 0] * 0.1
    Y[4, 4] = Y[4, 4] * 0.1

    # Ordinal date
    ord_x = np.array(map(dt.toordinal, x))

    # Create dataframe
    df = pd.DataFrame(np.vstack((Y, make_X(ord_x))).T,
                      columns=['b1', 'b2', 'b3', 'b4', 'b5', 'b7', 'b6'] +
                      ['B' + str(i) for i in range(8)],
                      index=x)

    print(ggplot(aes('B1', 'b5'), df) + geom_point() + ylim(0, 6000))

    X = make_X(ord_x).T

    yatsm = YATSM(X, Y)
    yatsm.run()

    import brewer2mpl
    ncolors = max(3, len(yatsm.record))
    colors = brewer2mpl.get_map('set1', 'qualitative', ncolors)
    breakpoints = yatsm.record['break']

    p = ggplot(aes('B1', 'b5'), df) + geom_point() + ylim(0, 6000)

    for i, r in enumerate(yatsm.record):
        ts_df = pd.DataFrame({ 'x': np.arange(r['start'], r['end']) })
        ts_df['y'] = np.dot(r['coef'][:, 4], make_X(ts_df['x']))

        p = p + geom_line(aes('x', 'y'), ts_df, color=colors.hex_colors[i])

        if r['break'] != 0:
            p = p + geom_vline(xintercept=r['break'], color='red')

    print(p)
