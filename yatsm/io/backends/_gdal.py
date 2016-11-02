""" Tools related to reading time series data using GDAL / rasterio
"""
from functools import partial
import logging
import os

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

logger = logging.getLogger(__name__)


def parse_dataset_file(input_file, date_format, column_dtype=None):
    """ Return parsed dataset CSV file as pd.DataFrame

    Args:
        input_file (str): CSV filename
        date_format (str): Format of date in input file
        column_dtype (dict): Datatype format parsing options for
            all or subset of columns passed as ``dtype`` argument to
            ``pandas.read_csv``

    Returns:
        pd.DataFrame: Dataset information
    """
    dt_parser = lambda x: pd.datetime.strptime(x, date_format)

    df = pd.read_csv(input_file,
                     parse_dates=['date'], date_parser=dt_parser,
                     dtype=column_dtype)
    df.set_index('date', inplace=True, drop=False)
    df.index.name = 'time'

    df.sort_values('date', inplace=True)

    if not os.path.isabs(df['filename'][0]):
        _root = os.path.abspath(os.path.dirname(input_file))
        df['filename'] = map(partial(os.path.join, _root), df['filename'])

    return df


class GDALTimeSeries(object):
    """ A time series that can be read in by GDAL

    Args:
        df (pd.DataFrame): A Pandas dataframe describing time series. Requires
            keys 'filename' and 'date' be column names. Additional columns
            will be used as metadata available via ``get_metadata``.
        input_file (str): If ``df`` is not specified, read time series dataset
            information from this file
        date_format (str): If ``df`` is not specified, parse date column in
            ``input_file`` with this date string format
        column_dtype (dict[str, str]): Datatype format parsing options for
            all or subset of ``df`` columns passed as ``dtype`` argument to
            ``pandas.read_csv``.
        keep_open (bool): Keep ``rasterio`` file descriptors open once opened?

    """
    def __init__(self, df=None, input_file='', date_format='%Y%m%d',
                 column_dtype=None,
                 keep_open=False, **kwargs):
        if isinstance(df, pd.DataFrame):
            if not all([k in df.keys() for k in ('date', 'filename')]):
                raise KeyError('pd.DataFrame passed should contain "date" and '
                               '"filename" keys')
            self.df = df
        elif input_file and date_format:
            self.df = parse_dataset_file(input_file, date_format, column_dtype)
        else:
            raise ValueError('Must specify either a pd.DataFrame or both'
                             '"input_file" and "date_format" arguments')
        self.keep_open = keep_open

        # Determine if input file has extra metadata
        self.extra_md = self.df.columns.difference(['date', 'filename'])

        self._init_attrs_from_file(self.df['filename'][0])

    def _init_attrs_from_file(self, filename):
        with rasterio.Env():
            with rasterio.open(filename, 'r') as src:
                #: dict: rasterio metadata of first file
                self.md = src.meta.copy()
                self.crs = src.crs
                self.transform = src.transform
                self.res = src.res
                self.ul = src.ul(0, 0)
                self.height = src.height
                self.width = src.width
                self.count = src.count
                self.length = len(self.df)
                self.block_windows = list(src.block_windows())
                self.nodatavals = src.nodatavals
                # We only use one datatype for reading -- promote to largest
                # if hetereogeneous
                self.dtype = src.dtypes[0]
                if not all([dt == self.dtype for dt in src.dtypes[1:]]):
                    logger.warning('GDAL reader cannot read multiple data '
                                   'types. Promoting memory allocation to '
                                   'largest datatype of source bands')
                    for dtype in src.dtypes[1:]:
                        self.dtype = np.promote_types(self.dtype, dtype)

    @property
    def time(self):
        return self.df['date']

    @property
    def _src(self):
        """ An optionally memoized generator on time series datasets
        """
        if self.keep_open:
            if not hasattr(self, '_src_open'):
                with rasterio.Env():
                    self._src_open = [rasterio.open(f, 'r') for
                                      f in self.df['filename']]
            for _src in self._src_open:
                yield _src
        else:
            with rasterio.Env():
                for f in self.df['filename']:
                    yield rasterio.open(f, 'r')

    def read(self, window=None, out=None):
        """ Read time series, usually inside of a specified window

        .. todo::

            Allow reading of a subset of bands (make like ``rasterio``)

        Args:
            window (tuple): A pair (tuple) of pairs of ints specifying
                the start and stop indices of the window rows and columns
            out (np.ndarray): A NumPy array of pre-allocated memory to read
                the time series into. Its shape should be::

                (len(observations), len(bands), len(rows), len(columns))

        Returns:
            np.ndarray: A NumPy array containing the time series data
        """
        # Parse geographic/projected coordinates from window query
        ((row_min, row_max), (col_min, col_max)) = window
        x_min, y_min = (col_min, row_max) * self.transform
        x_max, y_max = (col_max, row_min) * self.transform
        coord_bounds = (x_min, y_min, x_max, y_max)

        shape = (self.length, self.count,
                 window[0][1] - window[0][0],
                 window[1][1] - window[1][0])
        if not isinstance(out, np.ndarray):
            # TODO: check `out` is compatible if provided by user
            logger.debug('Allocating memory to read data of shape {}'
                         .format(shape))
            out = np.empty((shape), dtype=self.dtype)

        for idx, src in enumerate(self._src):
            _window = src.window(*coord_bounds, boundless=True)
            src.read(window=_window, out=out[idx, ...], boundless=True)

        return out

    def read_dataarray(self, window=None,
                       name=None, band_names=None,
                       out=None):
        """ Read time series, usually inside of a window, as xarray.DataArray

        Args:
            window (tuple): A pair (tuple) of pairs of ints specifying
                the start and stop indices of the window rows and columns
            name (str): Name of the xr.DataArray
            band_names (list[str]): Names of bands to use for xarray.DataArray
            out (np.ndarray): A NumPy array of pre-allocated memory to read
                the time series into. Its shape should be::

                ((len(observations), len(bands), len(rows), len(columns))

        Returns:
            xarray.DataArray: A DataArray containing the time series data with
            coordinate dimenisons (time, band, y, and x)

        Raises:
            IndexError: if `band_names` is specified but is not the same length
                as the number of bands, `self.count`
        """
        if not band_names:
            pad = len(str(self.count))
            band_names = ['Band_' + str(b).zfill(pad) for b in
                          range(1, self.count + 1)]

        if band_names and len(band_names) != self.count:
            raise IndexError('{0.__class__.__name__} has {0.count} bands but '
                             '`band_names` provided has {1} names'
                             .format(self, len(band_names)))

        values = self.read(window=window, out=out)
        coords_y, coords_x = self.window_coords(window)

        da = xr.DataArray(
            values,
            name=name,
            dims=['time', 'band', 'y', 'x'],
            coords=[self.df['date'], band_names, coords_y, coords_x]
        )

        da.attrs['crs'] = self.crs.to_string()
        da.attrs['crs_wkt'] = self.crs.wkt
        da.attrs['transform'] = self.transform
        da.attrs['rs'] = self.res
        da.attrs['nodata'] = self.nodatavals

        return da

    def get_metadata(self, items=None):
        """ Return a xr.Dataset of metadata from the input image list

        Args:
            items (iterable): Subset of metadata column names (`self.extra_md`)
                to return

        Returns:
            xarray.Dataset: A Dataset containing the time series metadata
            with coordinate dimenisons (time)

        """
        if not items:
            items = self.extra_md
        return xr.Dataset.from_dataframe(self.df[items])

    def window_coords(self, window):
        """ Return Y/X coordinates of a raster to pass to xarray

        Args:
            window (tuple): Window to read from ((ymin, ymax), (xmin, xmax)) in
                pixel space

        Returns:
            tuple (np.ndarray, np.ndarray): Y and X coordinates for window
        """
        coord_y = self.ul[0] + self.res[0] * np.arange(*window[0])
        coord_x = self.ul[1] + self.res[1] * np.arange(*window[1])

        return (coord_y, coord_x)
