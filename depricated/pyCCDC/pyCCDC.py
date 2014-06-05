#!/usr/bin/env python
# coding: utf-8

# In[7]:

import math

import numpy as np
import statsmodels.api as sm

from lasso_ccdc import CCDCLasso
from ts_driver.timeseries_ccdc import CCDCTimeSeries, py2mldate, ml2pydate

from matplotlib import pylab as plt

from ts_driver.timeseries import ml2pydate

import ipdb

p = '/home/ceholden/Documents/landsat_stack/p022r049/images'

ts = CCDCTimeSeries(p, image_pattern='L*')
ts.set_px(26)
ts.set_py(12)
ts.get_ts_pixel(use_cache=False, do_cache=False)

consecutive = 3
n_times = 1.5
threshold = 1.96
n_coef = 8
min_years = 1
threshold_noise = 400
min_rmse = 100

ndays = 365.25

n_band = 7
green = 1
swir1 = 4

# Lasso parameters
kwargs = {
          'lambdas'    : (20, ),
          'standardize': True,
          }
alpha = 1
rho = 1

# test vs fit bands
fit_bands = np.arange(0, 7)
test_bands = np.arange(0, 7)

class CCDCRecord(object):

    def __init__(self):
        self.t_start = 0
        self.t_end = 0
        self.t_break = 0
        self.pos = 0
        self.coefs = np.zeros((n_coef, n_band))
        self.rmse = np.zeros(n_band)

    def __str__(self):
        return '\n'.join(k + ':\n' + str(v) for k, v in self.__dict__.iteritems())

n_record = 0
record = []
record.append(CCDCRecord())


# In[3]:

Y = ts.get_data(mask=True)
x = np.ma.masked_array([py2mldate(_d) for _d in ts.dates],
                       mask=Y[0, :].mask)
Y = np.ma.compress_cols(Y)
x = x.compressed()

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

X = make_X(x)


# In[4]:

def multitemp_mask(x, Y, n_year, crit=400):
    """ Multi-temporal cloud/shadow masking using RLM

    Returns np.array of booleans. False indicate failed mask test and should be masked
    """
    n_year = np.ceil(n_year)

    w = 2.0 * np.pi / ndays

    X = np.array([
                  np.ones_like(x),
                  np.cos(w * x),
                  np.sin(w * x),
                  np.cos(w / n_year * x),
                  np.sin(w / n_year * x)])

    green_RLM = sm.RLM(Y[green, :], X.T,
                       M=sm.robust.norms.TukeyBiweight())
    swir1_RLM = sm.RLM(Y[swir1, :], X.T,
                       M=sm.robust.norms.TukeyBiweight())

    return np.logical_and(green_RLM.fit().resid < crit,
                         swir1_RLM.fit().resid > -crit)


# In[8]:

# CCDC data
Y = ts.get_data(mask=True)

print 'First Y date: '
print Y[:, 0]

x = np.ma.masked_array([py2mldate(_d) for _d in ts.dates],
                       mask=Y[0, :].mask)
Y = np.ma.compress_cols(Y)[fit_bands, :]
x = x.compressed()
X = make_X(x)

n_record = 0
record = []
record.append(CCDCRecord())

# start with minimum required number of clear observations
here = n_times * n_coef - 1
here_ = here

# test if we have enough observations
if len(x) < (here + consecutive):
    print 'Not enough observations'
else:
    # Starting position - first observation
    start = 0
    # boolean for training period
    b_train = True
    # number of recorded time segments
    n_record = n_record + 1

# Loop until we hit end of series minus consecutive
_i = 0
while here <= len(x) - consecutive:
    _i += 1
    print '<==================== Loop ' + str(_i)
    # span in index
    span_index = here - start + 1
    # span in time
    span_time = (X[1, here] - X[1, start]) / ndays
    print 'span_time: ' + str(span_time)
    print 'here: ' + str(here)
    print 'start: ' + str(start)
    print 'span_index: ' + str(span_index)

    # If we've spanned enough observations and time
    if span_index >= n_times * n_coef and span_time >= min_years:
        ### Train model
        if b_train is True:
            # Remove noise (consecutive + 1 since we're using arange)
            index = np.arange(start, here + consecutive + 1, dtype=np.uint16)
            mask = multitemp_mask(X[1, index], Y[:, index], span_time, crit=threshold_noise)

            # update span in index after noise removal
            span_index = mask[:-consecutive].sum()

            # check if we have enough observations after noise removal
            if span_index < n_times * n_coef:
                print 'Not enough clear obs...' + str(span_index)
                get_ipython().magic(u'debug')
                # move to next observation
                here += 1
                continue

            # copy X and Y with noise removed
            cX = np.delete(X, index[np.where(~mask)], axis=1)
            cY = np.delete(Y, index[np.where(~mask)], axis=1)

            # record our current position - important for next iteration of noise removal
            here_ = here
            print 'here_: ' + str(here_)

            # update our position after noise removal
            print 'updating here'
            here = start + span_index - 1
            print '    size: ' + str(np.zeros(250)[start:here].shape)
            # update time after noise removal
            span_time = (cX[1, here] - cX[1, start]) / ndays

            print 'span_time {t}'.format(t=span_time)

            # has enough time passed?
            if span_time < min_years:
                # keep current position
                here = here_
                # iterate forward 1 step
                here += 1
                continue

            ### fit model
            # init model testing variables
            v_slope = [] # normalized slope
            v_start = [] # difference of the first obs
            v_end = [] # difference of the last clear obs

            # fit all bands
            lasso_fits = []
            for b in test_bands:
                # print 'FITTING: here=' + str(here) + ' band=' + str(b)
                # fit lasso model (here + 1 to include here'th value)
                lasso = CCDCLasso(cX[:, start:(here + 1)].T,
                                  cY[b, start:(here + 1)],
                                  alpha, rho)
                lasso.fit(**kwargs)

                # prevent ideal fit
                rmse = max(lasso.rmse(), min_rmse)

                # "anormalized slope values"
                normal_s = rmse / ndays
                # "normalized slope values"
                ratio_s = np.abs(lasso.coef()[1]) / float(normal_s)
                # print 'ratio_s: ' + str(ratio_s)
                # "max normalized slope values"
                v_slope.append(max(ratio_s, 0))
#                print 'v_slope: ' + str(v_slope[b])

                # compare first clear observation
                v_start.append(np.abs(cY[b, start] - lasso.fittedvalues()[0]) /
                               (threshold * rmse))
#                print 'v_start: ' + str(v_start[b])
                # compare last observation
                v_end.append(np.abs(cY[b, here] - lasso.fittedvalues()[-1]) /
                             (threshold * rmse))
#                print 'v_end: ' + str(v_end[b])


                lasso_fits.append(lasso)

#            print 'v_slope: '
#            print v_slope, sum(v_slope)
#            print 'v_start: '
#            print v_start, sum(v_start)
#            print 'v_end: '
#            print v_end, sum(v_end)

            # find stable start for each curve
            if np.max(v_slope) > 1 or np.mean(v_start) > 1 or np.mean(v_end) > 1:
                # start from next clear obs
                start += 1
                # keep original position
                here = here_
                # move forward to next clear observation
                here += 1
                continue
            else:
                # model ready
                b_train = False
                # count difference of i for each iteration
                # (prevents fitting each iteration - see below)
                count = 0
                # make removal of noise permanent
                X = cX
                Y = cY
                # clean up copies
                del(cX)
                del(cY)

                print 'Entering monitoring period! (here=' + str(here) + ')'

        if b_train is False:
            # ready to enter monitoring period
            index = np.arange(start, here)

#            if (here - start + 1) - count >= (here - start + 1 ) / 3.0:
            if X[1, here] - X[1, start] >= count + ndays:
                # store models and rmse
                lasso_fits = []
                rmse = []

                for b in fit_bands:
                    lasso = CCDCLasso(X[:, start:here].T,
                                      Y[b, start:here],
                                      alpha, rho)
                    lasso.fit(**kwargs)
                    lasso_fits.append(lasso)
                    rmse.append(lasso.rmse)

                # update fit iteration count
                count = here - start + 1

                # update record information
                # n_record - 1 to account for 0 indexing vs. len([one record]) = 1
                record[n_record - 1].t_start = X[1, start]
                record[n_record - 1].t_end = X[1, here]
                record[n_record - 1].t_break = 0
                record[n_record - 1].pos = np.NAN #TODO
                record[n_record - 1].coefs = np.array([_lasso.coef() for _lasso in lasso_fits])
                record[n_record - 1].rmse = np.array([_lasso.rmse() for _lasso in lasso_fits])
            else:
                # update information without iteration
                # record time of curve end
                record[n_record - 1].t_end = X[1, here]

            ### temporally changing / dynamic rmse
            # get DOY for each X
            doy_x = np.array([int(ml2pydate(_d).strftime('%j')) for _d in X[1, start:here]])
            # get DOY for for each consecutive
            doy_test = np.array([int(ml2pydate(_d).strftime('%j')) for _d in X[1, here:(here + consecutive)]])
            # get difference
            doy_diff = np.array([abs(_d - doy_x) for _d in doy_test])
            # find smallest indexes for n_times * n_coef
            doy_index = np.argsort(doy_diff, axis=1)[:, 0:(n_times * n_coef)]

            # calculate "Z-score" criteria for each test band
            z_score = np.zeros((consecutive, len(test_bands)), dtype=np.float32)
            for i in range(consecutive):
                for b in test_bands:
                    # rmse => residual (of nearest DOY obs) scaled by size**0.5
                    _Y = Y.take(doy_index[i], axis=1)[b]
#                    print doy_index[i]
#                    print lasso_fits[b].fittedvalues()[doy_index[i]]
                    # print doy_index[i]
                    _Yhat = lasso_fits[b].fittedvalues().take(doy_index[i])
                    resid = np.linalg.norm(_Y - _Yhat)
                    rmse = resid / math.sqrt(n_times * n_coef)

#                    rmse = ((Y.take(doy_index[i], axis=1)[b] -
#                             lasso_fits[b].predict(X.take(doy_index[i], axis=1))) /
#                            sqrt(len(doy_index)))

                    rmse = max(rmse, min_rmse)

                    z_score[i, b] = np.abs(Y[b, here + i] -
                        lasso_fits[b].predict(X[:, here + i])) / rmse

            # ipdb.set_trace()

            z_score_mean = z_score.mean(axis=1)

            if min(z_score_mean) > threshold:
                ipdb.set_trace()
                print 'Break!'
                # TODO
            elif z_score_mean[0] > threshold:
                ipdb.set_trace()
                # test observation is probably noise - delete it
                # TODO
                # stay and check again after noise removal
                here -= 1
                pass

    # continue to next observation
    here += 1


# In[ ]:

for r in record:
    b5coefs = r.coefs[4, :]

    plt_x = np.arange(r.t_start, r.t_end, 1)
    plt_X = make_X(plt_x)
    pred = np.array([np.dot(b5coefs, _X) for _X in plt_X.T])

    plt.plot(plt_x, pred)

plt.plot(X[1, :], Y[4, :], 'ro')
plt.show()

