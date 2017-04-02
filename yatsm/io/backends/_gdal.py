""" Tools related to reading time series data using GDAL / rasterio
"""
import datetime as dt
import logging
from pathlib import Path
import six

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import xarray as xr

from yatsm import __version__
from yatsm.gis import (BoundingBox,
                       georeference_variable,
                       make_xarray_coords,
                       make_xarray_crs,
                       window_coords as _window_coords)
from yatsm.gis.conventions import CF_NC_ATTRS
from yatsm.utils import np_promote_all_types

logger = logging.getLogger(__name__)


BLOCK_SHAPE_WARNING = ('Bands in "{f}" do not have the same block shapes. '
                       'Reading will be very slow unless you re-process the '
                       'to have uniform block shapes.')


def parse_dataset_file(input_file, date_column, date_format,
                       column_dtype=None):
    """ Return parsed dataset CSV file as pd.DataFrame

    Args:
        input_file (str): CSV filename
        date_column (str): Column containing datetime information
        date_format (str): Format of date in input file
        column_dtype (dict): Datatype format parsing options for
            all or subset of columns passed as ``dtype`` argument to
            ``pandas.read_csv``

    Returns:
        pd.DataFrame: Dataset information
    """
    def _parser(x):
        return pd.datetime.strptime(x, date_format)

    df = pd.read_csv(input_file,
                     parse_dates=[date_column],
                     date_parser=_parser,
                     dtype=column_dtype)
    df.set_index(date_column, inplace=True, drop=False)
    df.index.name = 'time'

    df.sort_values(date_column, inplace=True)
    df.rename(columns={date_column: 'date'}, inplace=True)

    # Handle relative paths
    root = Path(input_file).parent.resolve()
    if not Path(df['filename'][0]).is_absolute():
        df['filename'] = [str(root.joinpath(f)) for f in df['filename']]

    return df


class GDALTimeSeries(object):
    """ A time series that can be read in by GDAL

    Args:
        df (pd.DataFrame): A Pandas dataframe describing time series. Requires
            keys 'filename' and 'date' be column names. Additional columns
            will be used as metadata available via ``get_metadata``.
        band_names (list[str]): List of names to call each raster band
        keep_open (bool): Keep ``rasterio`` file descriptors open once opened?

    Raises:
        TypeError: If the ``df`` is not a :ref:`pd.DataFrame`
        KeyError: If the ``df`` does not contain "date" and "filename" keys
    """

    _SRC_KEY = '_src'

    def __init__(self, df, band_names=None, keep_open=False):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('Must provide a pandas.DataFrame')
        if not all([k in df.keys() for k in ('date', 'filename')]):
            raise KeyError('pd.DataFrame passed should contain "date" and '
                           '"filename" keys')
        self.df = df
        self.band_names = band_names
        self.keep_open = keep_open

        # Determine if input file has extra metadata
        self.extra_md = self.df.columns.difference(['date', 'filename'])
        self._init_attrs_from_file(self.df['filename'][0])

    @classmethod
    def from_config(cls, input_file, date_column='date', date_format='%Y%m%d',
                    column_dtype=None, **kwds):
        """ Init time series dataset from file, as used by config

        Args:
            input_file (str): Filename of file containing time series
                information to parse using :ref:`pandas.read_csv`
            date_column (str): Column containing datetime information
            date_format (str): If ``df`` is not specified, parse date column in
                ``input_file`` with this date string format
            column_dtype (dict[str, str]): Datatype format parsing options for
                all or subset of ``df`` columns passed as ``dtype`` argument to
                ``pandas.read_csv``.
            **kwds (dict): Options to pass to ``__init__``

        """
        df = parse_dataset_file(input_file,
                                date_column=date_column,
                                date_format=date_format,
                                column_dtype=column_dtype)
        return cls(df, **kwds)

    def _init_attrs_from_file(self, filename):
        with rasterio.Env():
            with rasterio.open(filename, 'r') as src:
                #: dict: rasterio metadata of first file
                self.md = src.meta.copy()
                self.crs = src.crs
                self.transform = src.transform
                self.bounds = src.bounds
                self.res = src.res
                self.ul = src.xy(0, 0, offset='ul')

                self.height = src.height
                self.width = src.width
                self.shape = src.shape
                self.count = src.count
                if not self.band_names:
                    pad = len(str(self.count))
                    self.band_names = [
                        'Band_' + str(b).zfill(pad)
                        for b in range(1, self.count + 1)]
                self.length = len(self.df)

                block_shapes = set(src.block_shapes)
                if len(block_shapes) != 1:
                    logger.warning(BLOCK_SHAPE_WARNING.format(f=filename))
                self.block_shapes = list(block_shapes)[0]
                self.block_windows = list(src.block_windows())

                self.nodatavals = src.nodatavals

                # We only use one datatype for reading -- promote to largest
                # if hetereogeneous
                self.dtype = np_promote_all_types(*src.dtypes)

    @property
    def time(self):
        return self.df['date']

    def _src(self, time=None):
        """ An optionally memoized generator on time series datasets
        """
        if self._SRC_KEY not in self.df and self.keep_open:
            self.df[self._SRC_KEY] = None  # init it as blank

        rows = self.df.loc[time] if time else self.df
        if isinstance(rows, pd.Series):
            rows = pd.DataFrame([rows])

        if self.keep_open:
            with rasterio.Env():  # TODO: pass options
                null = rows.index[rows[self._SRC_KEY].isnull()]

                self.df.loc[null, self._SRC_KEY] = [
                    rasterio.open(f, 'r') for f in
                    self.df.loc[null, 'filename']
                ]
            for ds in rows[self._SRC_KEY]:
                yield ds
        else:
            with rasterio.Env():  # TODO: pass options
                for f in rows['filename']:
                    yield rasterio.open(f, 'r')

    def read(self, indexes=None, window=None, time=None, out=None):
        """ Read time series, usually inside of a specified window

        .. todo::

            Maybe pass a scheduler argument and some number of processes
            to use?

        Args:
            indexes (list[int] or int): One or more band numbers to retrieve.
                If a `list`, returns a 3D array; otherwise a 2D
            window (rasterio.windows.Window): A pair (tuple) of pairs of
                ints specifying the start and stop indices of the window rows
                and columns
            time (slice): Time period slice
            out (np.ndarray): A NumPy array of pre-allocated memory to read
                the time series into. Its shape should be::

                (len(observations), len(bands), len(rows), len(columns))

        Returns:
            np.ndarray: A NumPy array containing the time series data
        """
        sources = list(self._src(time=time))
        length = len(sources)
        if list(indexes):
            n_band = len(indexes if isinstance(indexes, (tuple, list))
                         else [indexes])
        else:
            n_band = self.count

        if list(indexes) == list(range(1, self.count + 1)):
            logger.debug('You asked for all bands in expected order, and '
                         'therefore it will be treated as `None` in '
                         '`src.read` call')
            indexes = None

        if window:
            # Parse geographic/projected coordinates from window query
            ((row_min, row_max), (col_min, col_max)) = window
            x_min, y_min = (col_min, row_max) * self.transform
            x_max, y_max = (col_max, row_min) * self.transform
            coord_bounds = (x_min, y_min, x_max, y_max)

            shape = (length, n_band,
                     window[0][1] - window[0][0],
                     window[1][1] - window[1][0])
        else:
            logger.debug('No window passed - calculating manually')
            shape = (length, n_band, ) + self.shape
            coord_bounds = self.bounds

        if not isinstance(out, np.ndarray):
            # TODO: check `out` is compatible if provided by user
            logger.debug('Allocating memory to read data of shape {}'
                         .format(shape))
            out = np.empty((shape), dtype=self.dtype)

        # TODO: rasterio doesn't support multiple dtypes yet, so
        #       either alert user or fix it ourselves with our wrapper
        for idx, src in enumerate(sources):
            _window = src.window(*coord_bounds, boundless=True)
            src.read(indexes=indexes,
                     out=out[idx, ...],
                     window=_window,
                     masked=True,
                     boundless=True)

        return out

    def read_dataarray(self, indexes=None, bands=None, window=None, time=None,
                       name=None, out=None, encoding=None):
        """ Read time series, usually inside of a window, as xarray.DataArray

        Args:
            indexes (list[int]): Band indexes of each raster to read
            bands (list[str]): An alternative to ``indexes``, provide a list
                of band names corresponding to :ref:`self.band_names`
            window (rasterio.windows.Window): A pair (tuple) of pairs of
                ints specifying the start and stop indices of the window rows
                and columns
            time (str, slice): A time or slice of time to subset the read
                with (using a subset on :ref:`self.df`)
            name (str): Name of the xr.DataArray
            out (np.ndarray): A NumPy array of pre-allocated memory to read
                the time series into. Its shape should be::

                ((len(observations), len(bands), len(rows), len(columns))
            encoding (dict): Optionally, pass encoding information to
                xarray.DataArray

        Returns:
            xarray.DataArray: A DataArray containing the time series data with
            coordinate dimenisons (time, band, y, and x)

        Raises:
            IndexError: if `band_names` is specified but is not the same length
            as the number of bands, `self.count`
        """
        if indexes:
            n_band = len(indexes if isinstance(indexes, (tuple, list))
                         else [indexes])
            band_names = [self.band_names[i] for i in indexes]
        elif bands:
            n_band = len(bands)
            indexes = [(self.band_names.index(band) + 1) for band in bands]
            band_names = bands
        else:
            indexes = list(range(1, self.count + 1))
            band_names = self.band_names

        if len(band_names) > self.count:
            raise IndexError('{0.__class__.__name__} has {0.count} bands but '
                             'you asked for {1}'
                             .format(self, n_band))
        if not window:
            window = self.window_extent

        dates = (self.df.loc[time, 'date'] if time is not None
                 else self.df['date'])
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates, name='dates')

        values = self.read(indexes=indexes, out=out, window=window, time=time)
        coords_y, coords_x = self.window_coords(window)
        crs = make_xarray_crs(self.crs)
        transform = rasterio.windows.transform(window, self.transform)

        da = xr.DataArray(
            values,
            name=name,
            dims=['time', 'band', 'y', 'x'],
            coords=[dates, band_names, coords_y, coords_x],
            encoding=encoding
        )
        # TODO: turn these steps into generic "georeference" xr
        da = da.assign_coords(crs=crs)

        da = georeference_variable(da, self.crs, transform)
        da.attrs.update(CF_NC_ATTRS)
        da.attrs['history'] = ('Created by YATSM v{0} at {1}.'
                               .format(__version__,
                                       dt.datetime.now().isoformat()))
        da.attrs['nodata'] = np.asarray(self.nodatavals)[np.array(indexes) - 1]
        # TODO: da.encoding
        # TODO: _FillValue, scale_factor, add_offset somewhere else (!)
        #       because _FillValue/etc are only 1 value per "variable",
        #       and I don't know what we'd do here if the `bands` in
        #       the DataArray had different _FillValue
        #       Probably better to pass as array under non-CF names (e.g.,
        #       nodata)
        # TODO: add chunksizes here? should be related to block_shapes
        # TODO: zlib, complevel, etc in to_netcdf function
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

    def encoding(self, indexes=None, bands=None, zlib=True, complevel=4):
        if isinstance(bands, six.string_types):
            bands = [bands]
        if indexes:
            bands = [self.band_names[i - 1] for i in indexes]
        if bands is None:
            bands = self.band_names

        encoding = {}
        for band in bands:
            encoding[band] = {
                'dtype': self.dtype,
                'complevel': complevel,
                'zlib': zlib,
                'chunksizes': (1, ) + self.block_shapes
            }
        return encoding

    @property
    def window_extent(self):
        """ rasterio.window.Window: Dataset extent window
        """
        return Window(0, 0, self.width, self.height)

    def window_coords(self, window=None):
        """ Return Y/X coordinates of a raster to pass to xarray

        Args:
            window (rasterio.windows.Window): A pair (tuple) of pairs of
                ints specifying the start and stop indices of the window rows
                and columns

        Returns:
            tuple (np.ndarray, np.ndarray): Y and X coordinates for window
        """
        y, x = _window_coords(window or self.window_extent, self.transform)
        return make_xarray_coords(y, x, self.crs)

    def window_bounds(self, window=None):
        """ Return coordinate bounds of a given window

        Args:
            window (rasterio.windows.Window): A pair (tuple) of pairs of
                ints specifying the start and stop indices of the window rows
                and columns

        Returns:
            BoundingBox: Window left, bottom, right, top (x_min, y_min, x_max,
            y_max)
        """
        return BoundingBox(*rasterio.windows.bounds(window or self.window_extent,
                                                    self.transform))
