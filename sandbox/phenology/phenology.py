""" Implementation of Eli Melaas' Landsat phenology algorithm """
from __future__ import division

from datetime import datetime as dt
import math

import numpy as np
import numpy.lib.recfunctions

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

Rstats = importr('stats')


def EVI(red, nir, blue):
    """ Return the Enhanced Vegetation Index for a set of np.ndarrays

    EVI is calculated as:

    .. math::
        2.5 * \\frac{(NIR - RED)}{(NIR + C_1 * RED - C_2 * BLUE + L)}

    where:
        - :math:`RED` is the red band
        - :math:`NIR` is the near infrared band
        - :math:`BLUE` is the blue band
        - :math:`C_1 = 6`
        - :math:`C_2 = 7.5`
        - :math:`L = 1`

    Note: bands must be given in float datatype from [0, 1]

    Args:
      red (np.ndarray): red band
      nir (np.ndarray): NIR band
      blue (np.ndarray): blue band

    Returns:
      np.ndarray: EVI

    """
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


def group_years(years, interval=3):
    """ Return integers representing sequential groupings of years

    Note: years specified must be sorted

    Args:
      years (np.ndarray): the year corresponding to each EVI value
      interval (int, optional): number of years to group together (default: 3)

    Returns:
      np.ndarray: integers representing sequential year groupings

    """
    n_groups = math.ceil((years.max() - years.min()) / interval)
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
      evi (np.ndarray): EVI values
      periods (np.ndarray): intervals of years to group and scale together
      qmin (float, optional): lower quantile for scaling (default: 10)
      qmax (float, optional): upper quantile for scaling (default: 90)

    Returns:
      np.ndarray: scaled EVI array

    """
    _evi = evi.copy()
    for u in np.unique(periods):
        index = np.where(periods == u)
        evi_min = np.percentile(evi[index], qmin)
        evi_max = np.percentile(evi[index], qmax)
        _evi[index] = (evi[index] - evi_min) / (evi_max - evi_min)

    return _evi


def CRAN_spline(x, y, spar=0.55):
    """ Return a prediction function for a smoothing spline from R

    Use `rpy2` package to fit a smoothing spline using "smooth.spline".

    Args:
      x (np.ndarray): independent variable
      y (np.ndarray): dependent variable
      spar (float): smoothing parameter

    Returns:
      callable: prediction function of smoothing spline that provides
        smoothed estimates of the dependent variable given an input
        independent variable array

    Example:
      Fit a smoothing spline for y ~ x and predict for days in year::

        pred_spl = CRAN_spline(x, y)
        y_smooth = pred_spl(np.arange(1, 366))

    """
    spl = Rstats.smooth_spline(x, y, spar=spar)

    return lambda _x: np.array(Rstats.predict_smooth_spline(spl, _x)[1])


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
    yeardoy[:, 0] = np.array([int(_d.strftime('%Y')) for _d in _date])
    yeardoy[:, 1] = np.array([int(_d.strftime('%j')) for _d in _date])

    return yeardoy


class LongTermMeanPhenology(object):
    """ Calculate long term mean phenology metrics for each YATSM record

    Long term mean phenology metrics describe the general spring greenup and
    autumn senescence timing using an algorithm by Melaas *et al.*, 2013 based
    on fitting smoothing splines to timeseries of EVI.

    Attributes:
      self.pheno (np.ndarray): NumPy structured array containing phenology
        metrics. These metrics include:
            spring_doy: the long term mean day of year of the start of spring
            autumn_doy: the long term mean day of year of the start of autumn
            pheno_cor: the correlation coefficient of the observed EVI and the
              smoothed prediction
            peak_evi: the highest smoothed EVI value within the year (maximum
                amplitude of EVI)
            peak_doy: the day of year corresponding to the peak EVI value
            spline_evi: the smoothing spline prediction of EVI for days of year
                between 1 and 365
            pheno_nobs: the number of observations used to fit the smoothing
                spline

    Args:
      model (yatsm.YATSM): instance of `yatsm.YATSM` that has been run for
        change detection
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

    """
    def __init__(self, model, red_index=2, nir_index=3, blue_index=0,
                 scale=0.0001, evi_index=None, evi_scale=None):
        self.model = model
        if evi_index:
            if not isinstance(evi_scale, float):
                raise ValueError('Must provide scale factor for EVI')
            self.evi = model.Y[evi_index, :] * evi_scale
        else:
            self.evi = EVI(model.Y[red_index, :] * scale,
                           model.Y[nir_index, :] * scale,
                           model.Y[blue_index, :] * scale)

        self.ordinal = model.X[:, 1].astype(np.uint32)
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
            ('spline_evi', 'f8', 365),
            ('pheno_nobs', 'u2')
        ])

    def _fit_record(self, evi, yeardoy, year_interval, q_min, q_max):
        # Calculate year-to-year groupings for EVI normalization
        periods = group_years(yeardoy[:, 0], year_interval)
        evi_norm = scale_EVI(evi, periods, qmin=q_min, qmax=q_max)

        # Pad missing DOY values (e.g. in winter) with 0's to improve
        # spline fit
        pad_start = np.arange(1, yeardoy[:, 1].min() + 1)
        pad_end = np.arange(yeardoy[:, 1].max(), 365 + 1)

        pad_doy = np.concatenate((yeardoy[:, 1], pad_start, pad_end))
        pad_evi_norm = np.concatenate((
            evi_norm,
            np.zeros_like(pad_start, dtype=evi.dtype),
            np.zeros_like(pad_end, dtype=evi.dtype)
        ))

        # Fit spline and predict EVI
        spl_pred = CRAN_spline(pad_doy, pad_evi_norm, spar=0.55)
        evi_smooth = spl_pred(np.arange(1, 366))

        # Check correlation
        pheno_cor = np.corrcoef(evi_smooth[yeardoy[:, 1] - 1], evi_norm)[0, 1]

        # Separate into spring / autumn
        peak_doy = np.argmax(evi_smooth)
        peak_evi = np.max(evi_smooth)
        evi_smooth_spring = evi_smooth[:peak_doy + 1]
        evi_smooth_autumn = evi_smooth[peak_doy + 1:]

        # Compute half-maximum of spring logistic for "ruling in" image dates
        # (points) for anamoly calculation
        # Note: we add + 1 to go from index (on 0) to day of year (on 1)
        ltm_spring = np.argmin(
            np.abs(
                (evi_smooth_spring - np.nanmin(evi_smooth_spring)) /
                (np.nanmax(evi_smooth_spring) - np.nanmin(evi_smooth_spring))
                - 0.5)
        ) + 1

        ltm_autumn = np.argmin(
            np.abs(
                (evi_smooth_autumn - np.nanmin(evi_smooth_autumn)) /
                (np.nanmax(evi_smooth_autumn) - np.nanmin(evi_smooth_autumn))
                - 0.5)
        ) + 1 + peak_doy + 1

        return (ltm_spring, ltm_autumn, pheno_cor,
                peak_evi, peak_doy, evi_smooth)

    def fit(self, year_interval=3, q_min=10, q_max=90):
        """ Fit phenology metrics for each time segment within a YATSM model

        Args:
          year_interval (int, optional): number of years to group together when
            normalizing EVI to upper and lower percentiles of EVI within the
            group (default: 3)
          q_min (float, optional): lower percentile for scaling EVI
            (default: 10)
          q_max (float, optional): upper percentile for scaling EVI
            (default: 90)

        Returns:
          np.ndarray: updated copy of YATSM model instance with phenology
            added into yatsm.record structured array

        """
        for i, _record in enumerate(self.model.record):
            # Subset variables to range of current record
            rec_range = np.where((self.ordinal >= _record['start']) &
                                 (self.ordinal <= _record['end']))[0]
            if rec_range.size == 0:
                continue

            _evi = self.evi[rec_range]
            _yeardoy = self.yeardoy[rec_range, :]

            # Fit and save results
            _result = self._fit_record(_evi, _yeardoy,
                                       year_interval, q_min, q_max)

            self.pheno[i]['spring_doy'] = _result[0]
            self.pheno[i]['autumn_doy'] = _result[1]
            self.pheno[i]['pheno_cor'] = _result[2]
            self.pheno[i]['peak_evi'] = _result[3]
            self.pheno[i]['peak_doy'] = _result[4]
            self.pheno[i]['spline_evi'][:] = _result[5]
            self.pheno[i]['pheno_nobs'] = rec_range.size

        return np.lib.recfunctions.merge_arrays(
            (self.model.record, self.pheno), flatten=True)


def long_term_mean_phenology(yatsm):
    """ Calculate mean phenology metrics for each record in `yatsm` timeseries

    TODO:
        - add args for parameterization of phenology calculation
            + band indices for red/nir/blue
            + quantiles for scaling
            + year grouping intervals
        - perhaps create class interface, and also add above parameters

    Args:
      yatsm (yatsm.YATSM): instance of `yatsm.YATSM` that has been run for
        change detection

    Returns:
      np.array: array of NumPy record arrays containing long term mean spring
        transition, smoothing spline correlation, and autumn transition

    """
    # DEBUG - #DELETE
    import matplotlib.pyplot as plt

    # Parameters to move
    year_interval = 3
    q_min = 10
    q_max = 90

    # Convert to year and DOY
    yeardoy = ordinal2yeardoy(yatsm.X[:, 1].astype(np.uint32))

    # Calculate and mask EVI
    evi = EVI(yatsm.Y[2, :] / 10000,
              yatsm.Y[3, :] / 10000,
              yatsm.Y[0, :] / 10000)

    valid_evi = np.where((evi >= 0) & (evi <= 1))[0]
    evi = evi[valid_evi]
    yeardoy = yeardoy[valid_evi]

    # Find groupings of years for EVI normalization
    periods = group_years(yeardoy[:, 0], interval=year_interval)

    evi_norm = scale_EVI(evi, periods, qmin=q_min, qmax=q_max)

    # DEBUG - #DELETE
    plt.scatter(yeardoy[:, 1], evi, color='k')
    plt.scatter(yeardoy[:, 1], evi_norm, color='r')
    plt.show()

    # Pad missing DOY values (e.g. in winter) with 0's to improve spline fit
    pad_start = np.arange(1, yeardoy[:, 1].min() + 1)
    pad_end = np.arange(yeardoy[:, 1].max(), 365 + 1)

    pad_doy = np.concatenate((yeardoy[:, 1], pad_start, pad_end))
    pad_evi_norm = np.concatenate((
        evi_norm,
        np.zeros_like(pad_start, dtype=evi.dtype),
        np.zeros_like(pad_end, dtype=evi.dtype)
    ))

    # Fit spline and predict EVI
    spl_pred = CRAN_spline(pad_doy, pad_evi_norm, spar=0.55)
    evi_smooth = spl_pred(np.arange(1, 366))

    # Check correlation
    Rsmooth = np.corrcoef(evi_smooth[doy - 1], evi_norm)[0, 1]

    # Separate into spring / autumn
    peak_value = np.argmax(evi_smooth)
    evi_smooth_spring = evi_smooth[:peak_value + 1]
    evi_smooth_autumn = evi_smooth[peak_value + 1:]

    # Compute half-maximum of spring logistic for "ruling in" image dates
    # (points) for anamoly calculation
    # Note: we add + 1 to go from index (on 0) to day of year (on 1)
    ltm_spring = np.argmin(
        np.abs(
            (evi_smooth_spring - np.nanmin(evi_smooth_spring)) /
            (np.nanmax(evi_smooth_spring) - np.nanmin(evi_smooth_spring))
            - 0.5)
    ) + 1

    ltm_autumn = np.argmin(
        np.abs(
            (evi_smooth_autumn - np.nanmin(evi_smooth_autumn)) /
            (np.nanmax(evi_smooth_autumn) - np.nanmin(evi_smooth_autumn))
            - 0.5)
    ) + 1 + peak_value + 1


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('evi_test.csv')

    yr = np.array(df['yr'])
    doy = np.array(df['doy'])
    prd = np.array(df['prd'])
    evi = np.array(df['evi_row'])

    # OVERRIDE PERIODS SINCE WE AUTOMATE IT
    # prd = group_years(yr, interval=3)

    valid_evi = np.where((evi >= 0) & (evi <= 1))[0]
    evi = evi[valid_evi]
    yr = yr[valid_evi]
    doy = doy[valid_evi]
    prd = prd[valid_evi]

    evi_norm = scale_EVI(evi, prd)
    np.testing.assert_almost_equal(evi_norm.max(), 1.69351, decimal=5,
                                   err_msg='Scaled EVI max is not correct')
    np.testing.assert_almost_equal(evi_norm.min(), -0.2385107, decimal=5,
                                   err_msg='Scaled EVI min is not correct')

    # Pad missing DOY values (e.g. in winter) with 0's to improve spline fit
    pad_start = np.arange(1, doy.min() + 1)
    pad_end = np.arange(doy.max(), 365 + 1)

    pad_doy = np.concatenate((doy, pad_start, pad_end))
    pad_evi_norm = np.concatenate((
        evi_norm,
        np.zeros_like(pad_start, dtype=evi.dtype),
        np.zeros_like(pad_end, dtype=evi.dtype)
    ))

    np.testing.assert_almost_equal(pad_evi_norm.sum(), 237.4103, decimal=4,
                                   err_msg='Padded EVI sum is not correct')

    # Fit spline and predict EVI
    spl_pred = CRAN_spline(pad_doy, pad_evi_norm, spar=0.55)
    evi_smooth = spl_pred(np.arange(1, 366))

    # Check correlation
    Rsmooth = np.corrcoef(evi_smooth[doy - 1], evi_norm)[0, 1]

    # Separate into spring / autumn
    peak_value = np.argmax(evi_smooth)
    evi_smooth_spring = evi_smooth[:peak_value + 1]
    evi_smooth_autumn = evi_smooth[peak_value + 1:]

    # Compute half-maximum of spring logistic for "ruling in" image dates
    # (points) for anamoly calculation
    # Note: we add + 1 to go from index (on 0) to day of year (on 1)
    ltm_spring = np.argmin(
        np.abs(
            (evi_smooth_spring - np.nanmin(evi_smooth_spring)) /
            (np.nanmax(evi_smooth_spring) - np.nanmin(evi_smooth_spring))
            - 0.5)
    ) + 1

    ltm_autumn = np.argmin(
        np.abs(
            (evi_smooth_autumn - np.nanmin(evi_smooth_autumn)) /
            (np.nanmax(evi_smooth_autumn) - np.nanmin(evi_smooth_autumn))
            - 0.5)
    ) + 1 + peak_value + 1

    # Test
    np.testing.assert_almost_equal(Rsmooth, 0.9595954, decimal=5,
                                   err_msg='Spline correlation is not correct')
    np.testing.assert_equal(ltm_spring, 138,
                            err_msg='Spring LTM is not correct')
    np.testing.assert_equal(ltm_autumn, 283,
                            err_msg='Autumn LTM is not correct')
