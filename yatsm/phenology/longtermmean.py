""" Implementation of Eli Melaas' Landsat phenology algorithm

See:
    Melaas, EK, MA Friedl, and Z Zhu. 2013. Detecting interannual variation in
    deciduous broadleaf forest phenology using Landsat TM/ETM+ data. Remote
    Sensing of Environment 132: 176-185.

"""
from __future__ import division

from collections import namedtuple
from datetime import datetime as dt
import logging
import math

import numpy as np
import numpy.lib.recfunctions
import pandas as pd

from ..regression.cran import CRAN_spline
from ..vegetation_indices import EVI

logger = logging.getLogger('yatsm')


#: tuple: Tuple containing the results of the long term mean phenology
#         transition date calculation
LongTermMeanPhenologyResults = namedtuple('LongTermMeanPhenologyResults', [
    'springDOY', 'autumnDOY', 'peakEVIDOY',
    'peakEVI', 'corrcoef', 'smoothEVI'
])


def group_years(years, interval=3):
    """ Return integers representing sequential groupings of years

    Note: years specified must be sorted

    Args:
        years (np.ndarray): the year corresponding to each EVI value
        interval (int, optional): number of years to group together
            (default: 3)

    Returns:
        np.ndarray: integers representing sequential year groupings

    """
    n_groups = math.ceil((years.max() - years.min()) / interval)
    if n_groups <= 1:
        return np.zeros_like(years, dtype=np.uint16)
    splits = np.array_split(np.arange(years.min(), years.max() + 1), n_groups)

    groups = np.zeros_like(years, dtype=np.uint16)
    for i, s in enumerate(splits):
        groups[np.in1d(years, s)] = i

    return groups


def scale_EVI(evi, periods, qmin=10, qmax=90):
    """ Returns EVI scaled to upper and lower quantiles

    Quantiles are calculated based on EVI within some year-to-year interval.
    As part of finding the quantiles, EVI values not within the (0, 1) range
    will be removed.

    Args:
        evi (pd.Series): EVI values
        periods (np.ndarray): intervals of years to group and scale together
        qmin (float, optional): lower quantile for scaling (default: 10)
        qmax (float, optional): upper quantile for scaling (default: 90)

    Returns:
        pd.Series: scaled EVI array

    """
    _evi = evi.copy()
    for u in np.unique(periods):
        index = periods == u
        evi_min = np.percentile(evi[index], qmin)
        evi_max = np.percentile(evi[index], qmax)
        _evi[index] = (evi[index] - evi_min) / (evi_max - evi_min)

    return _evi


def halfmax(x):
    """ Return index of the observation closest to the half of some data

    Assumes that data are scaled between [0, 1] and half-max is 0.5

    Args:
        x (np.ndarray): a one dimensional vector

    Returns:
        int: the index of the observation closest to the half-max of the data

    """
    return np.argmin(np.abs(
        (x - np.nanmin(x)) /
        (np.nanmax(x) - np.nanmin(x)) - 0.5))


# TODO: delete
def ordinal2yeardoy(ordinal):
    """ Convert ordinal dates to two arrays of year and doy

    Args:
        ordinal (np.ndarray): ordinal dates

    Returns:
        np.ndarray: nobs x 2 np.ndarray containing the year and DOY for each
            ordinal date

    """
    _date = [dt.fromordinal(_d) for _d in ordinal]
    yeardoy = np.empty((ordinal.size, 2), dtype=np.uint16)
    yeardoy[:, 0] = np.array([_d.timetuple().tm_year for _d in _date])
    yeardoy[:, 1] = np.array([_d.timetuple().tm_yday for _d in _date])

    return yeardoy


def longtermmeanphenology(evi, periods=None, year_interval=3,
                          q_min=10., q_max=90.):
    """ Calculate the long term mean phenology transition dates

    EVI should be scaled within [0, 1]. Values outside of this range will be
    removed before processing.

    Args:
        evi (pd.Series): EVI values in a Pandas Series. The index should be
            the date of each observation
        periods (pd.Series): User provided groupings to scale EVI time series
            with. If not provided, groupings will be calculated based on
            ``year_interval``.
        year_interval (int): number of years to group together when
            normalizing EVI to upper and lower percentiles of EVI within the
            group
        q_min (float): lower percentile for scaling EVI
        q_max (float): upper percentile for scaling EVI

    Returns:
        LongTermMeanPhenologyResults: Named tuple of phenological transition
            dates and diagnostics

    """
    evi = evi.loc[evi.notnull() & (evi >= 0.) & (evi <= 1.)]

    # Calculate year-to-year groupings for EVI normalization
    if periods is None:
        periods = group_years(evi.index.year, year_interval)
    else:
        periods = periods[evi.index]
    evi_norm = scale_EVI(evi, periods, qmin=q_min, qmax=q_max)

    # Mask out np.nan
    evi_norm = evi_norm.loc[evi_norm.notnull()]
    if evi_norm.size == 0:
        logger.debug('No valid EVI after scaling -- skipping')
        return

    # Pad missing DOY values (e.g. in winter) with 0's to improve
    # spline fit
    def make_date(doy):
        return pd.DatetimeIndex(['2000']) + doy.astype('timedelta64[D]')
    pad_start = make_date(np.arange(1, evi_norm.index.dayofyear.min() + 1))
    pad_end = make_date(np.arange(evi_norm.index.dayofyear.max(), 365 + 1))

    pad_evi_norm = pd.concat((
        evi_norm,
        pd.Series(np.zeros(pad_start.size), index=pad_start),
        pd.Series(np.zeros(pad_end.size), index=pad_end)
    ))

    # Fit spline and predict EVI
    spl_pred = CRAN_spline(pad_evi_norm.index.dayofyear, pad_evi_norm.values,
                           spar=0.55)
    # 366 to include leap years
    evi_smooth = pd.Series(spl_pred(np.arange(1, 367)),
                           index=np.arange(1, 367))

    # Check correlation
    pheno_cor = np.corrcoef(evi_smooth[evi_norm.index.dayofyear],
                            evi_norm)[0, 1]

    # Separate into spring / autumn
    peak_doy = evi_smooth.argmax()
    peak_evi = evi_smooth.max()
    evi_smooth_spring = evi_smooth[:peak_doy + 1]
    evi_smooth_autumn = evi_smooth[peak_doy + 1:]

    # Compute half-maximum of spring logistic for "ruling in" image dates
    # (points) for anomaly calculation
    # Note: we add + 1 to go from index (on 0) to day of year (on 1)
    if evi_smooth_spring.size > 0:
        ltm_spring = halfmax(evi_smooth_spring)
    else:
        ltm_spring = 0

    if evi_smooth_autumn.size > 0:
        ltm_autumn = halfmax(evi_smooth_autumn)
    else:
        ltm_autumn = 0

    return LongTermMeanPhenologyResults(
        springDOY=ltm_spring, autumnDOY=ltm_autumn, peakEVIDOY=peak_doy,
        peakEVI=peak_evi, corrcoef=pheno_cor, smoothEVI=evi_smooth
    )


class LongTermMeanPhenology(object):
    """ Calculate long term mean phenology metrics for each YATSM record

    Long term mean phenology metrics describe the general spring greenup and
    autumn senescence timing using an algorithm by Melaas *et al.*, 2013 based
    on fitting smoothing splines to timeseries of EVI.

    Attributes:
        self.pheno (np.ndarray): NumPy structured array containing phenology
            metrics. These metrics include:

            * spring_doy: the long term mean day of year of the start of spring
            * autumn_doy: the long term mean day of year of the start of autumn
            * pheno_cor: the correlation coefficient of the observed EVI and
              the smoothed prediction
            * peak_evi: the highest smoothed EVI value within the year (maximum
              amplitude of EVI)
            * peak_doy: the day of year corresponding to the peak EVI value
            * spline_evi: the smoothing spline prediction of EVI for days of
              year between 1 and 365
            * pheno_nobs: the number of observations used to fit the smoothing
              spline

    Args:
        red_index (int, optional): index of model.Y containing red band
            (default: 2)
        nir_index (int, optional): index of model.Y containing NIR band
            (default: 3)
        blue_index (int, optional): index of model.Y containing blue band
            (default: 0)
        scale (float or np.ndarray, optional): scale factor for reflectance
            bands in model.Y to transform data into [0, 1] (default: 0.0001)
        evi_index (int, optional): if EVI is already used within timeseries
            model, provide index of model.Y containing EVI to override
            computation from red/nir/blue bands (default: None)
        evi_scale (float, optional): if EVI is already used within timeseries
            model, provide scale factor to transform EVI into [0, 1] range
            (default: None)
        year_interval (int, optional): number of years to group together when
            normalizing EVI to upper and lower percentiles of EVI within the
            group (default: 3)
        q_min (float, optional): lower percentile for scaling EVI (default: 10)
        q_max (float, optional): upper percentile for scaling EVI (default: 90)

    """
    def __init__(self, red_index=2, nir_index=3, blue_index=0,
                 scale=0.0001, evi_index=None, evi_scale=None,
                 year_interval=3, q_min=10, q_max=90):
        self.red_index = red_index
        self.nir_index = nir_index
        self.blue_index = blue_index
        self.scale = scale
        self.evi_index = evi_index
        self.evi_scale = evi_scale
        self.year_interval = year_interval
        self.q_min = q_min
        self.q_max = q_max

    def _fit_prep(self, model):
        if self.evi_index:
            if not isinstance(self.evi_scale, float):
                raise ValueError('Must provide scale factor for EVI')
            self.evi = model.Y[self.evi_index, :] * self.evi_scale
        else:
            self.evi = EVI(model.Y[self.red_index, :] * self.scale,
                           model.Y[self.nir_index, :] * self.scale,
                           model.Y[self.blue_index, :] * self.scale)

        self.ordinal = model.dates.astype(np.uint32)
        self.yeardoy = ordinal2yeardoy(self.ordinal)

        # Mask based on unusual EVI values
        valid_evi = np.where((self.evi >= 0) & (self.evi <= 1))[0]
        self.evi = self.evi[valid_evi]
        self.ordinal = self.ordinal[valid_evi]
        self.yeardoy = self.yeardoy[valid_evi, :]

        self.pheno = np.zeros(self.model.record.shape, dtype=[
            ('spring_doy', 'u2'),
            ('autumn_doy', 'u2'),
            ('pheno_cor', 'f4'),
            ('peak_evi', 'f4'),
            ('peak_doy', 'u2'),
            ('spline_evi', 'f8', 366),
            ('pheno_nobs', 'u2')
        ])


    def fit(self, model):
        """ Fit phenology metrics for each time segment within a YATSM model

        Args:
            model (yatsm.YATSM): instance of `yatsm.YATSM` that has been run
                for change detection

        Returns:
            np.ndarray: updated copy of YATSM model instance with phenology
                added into yatsm.record structured array

        """
        self.model = model
        # Preprocess EVI and create our `self.pheno` record
        self._fit_prep(model)

        for i, _record in enumerate(self.model.record):
            # Subset variables to range of current record
            rec_range = np.where((self.ordinal >= _record['start']) &
                                 (self.ordinal <= _record['end']))[0]
            if rec_range.size == 0:
                continue

            _evi = self.evi[rec_range]
            _yeardoy = self.yeardoy[rec_range, :]

            # TODO
            # Fit and save results
            _result = self._fit_record(_evi, _yeardoy,
                                       self.year_interval,
                                       self.q_min, self.q_max)
            if _result is None:
                continue

            self.pheno[i]['spring_doy'] = _result[0]
            self.pheno[i]['autumn_doy'] = _result[1]
            self.pheno[i]['pheno_cor'] = _result[2]
            self.pheno[i]['peak_evi'] = _result[3]
            self.pheno[i]['peak_doy'] = _result[4]
            self.pheno[i]['spline_evi'][:] = _result[5]
            self.pheno[i]['pheno_nobs'] = rec_range.size

        return np.lib.recfunctions.merge_arrays(
            (self.model.record, self.pheno), flatten=True)
