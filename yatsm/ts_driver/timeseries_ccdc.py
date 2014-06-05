# -*- coding: utf-8 -*-
# vim: set expandtab:ts=4
"""
/***************************************************************************
 CCDCTimeSeries
                                 A QGIS plugin
 Plotting & visualization tools for CCDC Landsat time series analysis
                             -------------------
        begin                : 2013-03-15
        copyright            : (C) 2013 by Chris Holden
        email                : ceholden@bu.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import datetime as dt
import fnmatch
import os
import sys

import numpy as np
import scipy.io
try:
    from osgeo import gdal
except:
    import gdal

import timeseries
from timeseries import mat2dict, ml2pydate, py2mldate

class CCDCTimeSeries(timeseries.AbstractTimeSeries):
    """Class holding data and methods for time series used by CCDC 
    (Change Detection and Classification). Useful for QGIS plugin 'TSTools'.

    More doc TODO
    """

    # __str__ name for TSTools data model plugin loader
    __str__ = 'CCDC Time Series'

    # TODO add some container for "metadata" that can be used in table
    image_names = []
    filenames = []
    filepaths = []
    length = 0
    dates = np.array([])
    n_band = 0
    _data = np.array([])
    _tmp_data = np.array([])
    result = []

    has_results = False
    
    x_size = 0
    y_size = 0
    geo_transform = None
    projection = None
    fformat = None
    datatype = None
    band_names = []
    readers = []

    _px = None
    _py = None

    image_pattern = 'LND*'
    stack_pattern = '*stack'
    results_folder = 'TSFitMap'
    results_pattern = 'record_change*'

    def __init__(self, location, 
                 image_pattern=image_pattern, 
                 stack_pattern=stack_pattern,
                 results_folder=results_folder,
                 results_pattern=results_pattern,
                 cache_folder='.cache'):
        
        super(CCDCTimeSeries, self).__init__(location, 
                                             image_pattern,
                                             stack_pattern)
        self.results_folder = results_folder
        self.results_pattern = results_pattern
        self.cache_folder = cache_folder

        self.mask_band = 7
        self.mask_val = [2, 3, 4, 255]

        self._find_stacks()
        self._get_attributes()
        self._get_dates()
        self._check_results()
        self._check_cache()
        self._open_ts()

        self._data = np.zeros([self.n_band, self.length], dtype=self.datatype)

    def get_ts_pixel(self, use_cache=True, do_cache=True):
        """ Fetch pixel data for the current pixel and set to self._data 
        
        Uses:
            self._px, self._py

        Args:
            use_cache               allow for retrieval of data from cache
            do_cache                enable caching of retrieved results
                 
        """
        read_data = False
        wrote_data = False

        if self.has_cache is True and use_cache is True:
            read_data = self.retrieve_from_cache()

        if read_data is False:
            # Read in from images
            for i in xrange(self.length):
                self.retrieve_pixel(i)

        # Apply mask
        self.apply_mask()

        # Try to cache result if we didn't just read it from cache
        if self.can_cache is True and do_cache is True and read_data is False:
            try:
                wrote_data = self.write_to_cache()
            except:
                print 'Could not write to cache file'
                raise
            else:
                if wrote_data is True:
                    print 'Wrote to cache file'


    def retrieve_pixel(self, index):
        """ Retrieve pixel data for a given x/y and index in time series
    
        Uses:
            self._px, self._py

        Args:
            index                   index of image in time series

        """
        if self._tmp_data.shape[0] == 0:
            self._tmp_data = np.zeros_like(self._data)

        # Read in from images
        self._tmp_data[:, index] = self.readers[index].get_pixel(
            self._py, self._px)
        self._data[:, index] = self.readers[index].get_pixel(
            self._py, self._px)
        
        if index == self.length - 1:
            print 'Last result coming in!'
            self._data = self._tmp_data
            self._tmp_data = np.array([])

    def retrieve_result(self):
        """ Returns the record changes for the current pixel

        Result is stored as a list of dictionaries

        Note:   MATLAB indexes on 1 so y is really (y - 1) in calculations and
                x is (x - 1)

        """
        self.result = []

        # Check for existence of output
        record = self.results_pattern.replace('*', str(self._py + 1)) + '.mat'
        record = os.path.join(self.location, self.results_folder, record)
        
        print 'Opening: {r}'.format(r=record)

        if not os.path.exists(record):
            print 'Warning: cannot find record for row {r}: {f}'.format(
                r=self._py + 1, f=record)
            return

        # Calculate MATLAB position for x, y
        pos = (self._py * self.x_size) + self._px + 1

        print '    position: {p}'.format(p=pos)

        # Read .mat file as ndarray of scipy.io.matlab.mio5_params.mat_struct
        mat = scipy.io.loadmat(record, squeeze_me=True, 
                               struct_as_record=False)['rec_cg']

        # Loop through to find correct x, y
        for i in xrange(mat.shape[0]):
            if mat[i].pos == pos:
                self.result.append(mat2dict(mat[i]))

    def get_data(self, mask=True):
        """ Return time series dataset with options to mask/unmask
        """
        if mask is False:
            return np.array(self._data)
        else:
            return self._data

    def get_prediction(self, band, usermx=None):
        """ Return the time series model fit predictions for any single pixel

        Arguments:
            band            time series band to predict
            usermx          optional - can specify MATLAB datenum dates as list

        Returns:
            [(mx, my)]      list of data points for time series fit where 
                                length of list is equal to number of time
                                segments

        """
        if usermx is None:
            has_mx = False
        else:
            has_mx = True
        mx = []
        my = []

        if len(self.result) > 0:
            for rec in self.result:
                if band >= rec['coefs'].shape[1]:
                    break
                
                ### Setup x values (dates)
                # Use user specified values, if possible
                if has_mx:
                    _mx = usermx[np.where((usermx >= rec['t_start']) & 
                                      (usermx <= rec['t_end']))]
                    if len(_mx) == 0: 
                        # User didn't ask for dates in this range
                        continue
                else:
                # Create sequence of MATLAB ordinal date
                    _mx = np.linspace(rec['t_start'],
                                      rec['t_end'],
                                      rec['t_end'] - rec['t_start'])
                coef = rec['coefs'][:, band]
                
                ### Calculate model predictions
                # HACK: adjust w based on existence of parameter file
                if os.path.isfile(os.path.join(
                    self.location, self.results_folder, '.p_36525')):
                    w = 2 * np.pi / 365.25
                else:
                    w = 2 * np.pi / 365

                if coef.shape[0] == 4:
                    # 4 coefficient model
                    _my = (coef[0] +
                            coef[1] * _mx +
                            coef[2] * np.cos(w * _mx) +
                            coef[3] * np.sin(w * _mx))
                elif coef.shape[0] == 6:
                    # 6 coefficient model
                    _my = (coef[0] + 
                           coef[1] * _mx + 
                           coef[2] * np.cos(w * _mx) + 
                           coef[3] * np.sin(w * _mx) +
                           coef[4] * np.cos(2 * w * _mx) +
                           coef[5] * np.sin(2 * w * _mx))
                elif coef.shape[0] == 8:
                    # 8 coefficient model
                    _my = (coef[0] +
                            coef[1] * _mx +
                            coef[2] * np.cos(w * _mx) +
                            coef[3] * np.sin(w * _mx) +
                            coef[4] * np.cos(2 * w * _mx) +
                            coef[5] * np.sin(2 * w * _mx) +
                            coef[6] * np.cos(3 * w * _mx) +
                            coef[7] * np.sin(3 * w * _mx))
                else:
                    break
                ### Transform MATLAB ordinal date into Python datetime
                _mx = [dt.datetime.fromordinal(int(m)) -
                                dt.timedelta(days = 366)
                                for m in _mx]
                ### Append
                mx.append(np.array(_mx))
                my.append(np.array(_my))

        return (mx, my)

    def get_breaks(self, band):
        """ Return an array of (x, y) data points for time series breaks """
        bx = []
        by = []
        if len(self.result) > 1:
            for rec in self.result[0:-1]:
                bx.append(dt.datetime.fromordinal(int(rec['t_break'])) -
                      dt.timedelta(days = 366))
                print 'Break: %s' % str(bx)
                index = [i for i, date in 
                        enumerate(self.dates) if date == bx[-1]][0]
                print 'Index: %s' % str(index)
                if index < self._data.shape[1]:
                    by.append(self._data[band, index])

        return (bx, by)

    def get_px(self):
        """ Returns current pixel column number """
        return self._px

    def set_px(self, x):
        """ Set current pixel column number """
        if x < 0:
            raise ValueError('x cannot be below 0')
        elif x > self.x_size:
            raise ValueError('x cannot be larger than the image')
        elif x is None:
            raise ValueError('x cannot be None')
        else:
            self._px = x

    def get_py(self):
        """ Returns current pixel row number """
        return self._py

    def set_py(self, y):
        """ Set current pixel row number """
        if y < 0:
            raise ValueError('y cannot be below 0')
        elif y > self.y_size:
            raise ValueError('y cannot be larger than the image')
        elif y is None:
            raise ValueError('y cannot be None')
        else:
            self._py = y

### OVERRIDEN "ADDITIONAL" OPTIONAL METHODS SUPPORTED BY CCDCTimeSeries
    def apply_mask(self, mask_band=None, mask_val=None):
        """ Apply mask to self._data """
        if mask_band is None:
            mask_band = self.mask_band
        if mask_val is None:
            mask_val = list(self.mask_val)

        self._data = np.array(self._data)

        mask = np.ones_like(self._data) * np.logical_or.reduce(
            [self._data[mask_band, :] == mv for mv in mask_val])

        self._data = np.ma.MaskedArray(self._data, mask=mask)

    def retrieve_from_cache(self):
        """ Try retrieving a pixel timeseries from cache 
        
        Return True, False or Exception depending on success

        """
        # Test caching for retrieval
        cache = self.cache_name_lookup(self._px, self._py) 
        if self.has_cache is True and os.path.exists(cache):
            try:
                _read_data = np.load(cache) 
            except:
                print 'Error: could not open pixel {x}/{y} from cache ' \
                        'file'.format(x=self._px, y=self._py)
                print sys.exc_info()[0]
                raise
            else:
                # Test if newly read data is same size as current
                if _read_data.shape != self._data.shape:
                    print 'Warning: cached data may be out of date'
                    return False

                self._data = _read_data

                # We've read data, apply mask and return True
                self.apply_mask()
                
                return True

        return False

    def write_to_cache(self):
        """ Write retrieved time series to cache

        Return True, False, or Exception depending on success

        Note:   writing of NumPy masked arrays is not implemented, so stick to 
                regular ndarray

        """
        cache = self.cache_name_lookup(self._px, self._py)

        # Try to cache result
        if self.has_cache is True and self.can_cache is True:
            try:
                np.save(cache, np.array(self._data))
            except:
                print 'Error: could not write pixel {x}/{y} to cache ' \
                        'file'.format(x=self._px, y=self._py)
                print sys.exc_info()[0]
                raise
            else:
                return True
        return False

### INTERNAL SETUP METHODS
    def _find_stacks(self):
        """ Find and set names for Landsat image stacks """
        # Setup lists
        self.image_names = []
        self.filenames = []
        self.filepaths = []

        # Populate - only checking one directory down
        self.location = self.location.rstrip(os.path.sep)
        num_sep = self.location.count(os.path.sep)
        for root, dnames, fnames in os.walk(self.location, followlinks=True):
            # Remove results folder
            dnames[:] = [d for d in dnames if self.results_folder not in d]

            # Force only 1 level
            num_sep_this = root.count(os.path.sep)
            if num_sep + 1 <= num_sep_this:
                del dnames[:]

            # Directory names as image IDs
            for dname in fnmatch.filter(dnames, self.image_pattern):
                self.image_names.append(dname)
            # Add file name and paths
            for fname in fnmatch.filter(fnames, self.stack_pattern):
                self.filenames.append(fname)
                self.filepaths.append(os.path.join(root, fname))

        # Check for consistency
        if len(self.image_names) != len(self.filenames) != len(self.filepaths):
            raise Exception(
                'Inconsistent number of stacks and stack directories')

        self.length = len(self.image_names)
        if self.length == 0:
            raise Exception('Zero stack images found')

        # Sort by image name/ID (i.e. Landsat ID)
        self.image_names, self.filenames, self.filepaths = (list(t) for t in
            zip(*sorted(zip(self.image_names, self.filenames, self.filepaths))))

    def _get_attributes(self):
        """ Fetch image stack attributes including number of rows, columns,
        bands, the geographic transform, projection, file format, data type,
        and band names
        
        """
        # Based on first stack image
        stack = self.filepaths[0]

        # Open with GDAL
        gdal.AllRegister()
        ds = gdal.Open(stack, gdal.GA_ReadOnly)
        if ds is None:
            raise Exception('Could not open {stack} as dataset'.format(
                stack=stack))

        # Raster size
        self.x_size = ds.RasterXSize
        self.y_size = ds.RasterYSize
        self.n_band = ds.RasterCount

        # Geographic transform & projection
        self.geo_transform = ds.GetGeoTransform()
        self.projection = ds.GetProjection()

        # File type and format
        self.fformat = ds.GetDriver().ShortName
        if self.fformat == 'ENVI':
            interleave = ds.GetMetadata('IMAGE_STRUCTURE')['INTERLEAVE']
            if interleave == 'PIXEL':
                self.fformat = 'BIP'
            elif interleave == 'BAND':
                self.fformat = 'BSQ'

        # Data type
        self.datatype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
        if self.datatype == 'Byte':
            self.datatype = 'uint8'
        self.datatype = np.dtype(self.datatype)

        # Band names
        for i in range(ds.RasterCount):
            band = ds.GetRasterBand(i + 1)
            if (band.GetDescription() is not None and 
                len(band.GetDescription()) > 0):
                self.band_names.append(band.GetDescription())
            else:
                self.band_names.append('Band %s' % str(i + 1))

        ds = None

    def _get_dates(self):
        """ Get image dates as Python datetime

        Note:   Because we're trying to get YEAR-DOY, we have to first get the
                year and DOY separately to create the date using:
                    datetime(year, 1, 1) + timedelta(doy - 1)

        """
        self.dates = []
        for image_name in self.image_names:
            self.dates.append(dt.datetime(int(image_name[9:13]), 1, 1) +
                              dt.timedelta(int(image_name[13:16]) - 1))
        self.dates = np.array(self.dates)

        # Sort images by date
        self.dates, self.image_names, self.filenames, self.filepaths = (
            list(t) for t in zip(*sorted(zip(
                self.dates, self.image_names, self.filenames, self.filepaths)))
        )
        self.dates = np.array(self.dates)

    def _check_results(self):
        """ Checks for results """
        results = os.path.join(self.location, self.results_folder)
        if (os.path.exists(results) and os.path.isdir(results) and 
            os.access(results, os.R_OK)):
            # Check for any results
            for root, dname, fname in os.walk(results):
                for f in fnmatch.filter(fname, self.results_pattern):
                    self.has_results = True
                    self.results_folder = root 
                    return

    def _check_cache(self):
        """ Checks location of time series for a cache to read/write 
        time series
        """
        cache = os.path.join(self.location, self.cache_folder)
        if os.path.exists(cache) and os.path.isdir(cache):
            if os.access(cache, os.R_OK):
                self.has_cache = True
            else:
                self.has_cache = False
            
            if os.access(cache, os.W_OK):
                self.can_cache = True
            else:
                self.can_cache = False
        else:
            try:
                os.mkdir(cache)
            except:
                pass
            else:
                self.has_cache = True
                self.can_cache = True

        print 'Has cache?: {b}'.format(b=self.has_cache)
        print 'Can cache?: {b}'.format(b=self.can_cache)

    def _open_ts(self):
        """ Open timeseries as list of CCDCBinaryReaders """
        self.readers = []
        if (self.fformat == 'ENVI' or self.fformat == 'BIP' or self.fformat == 
                'BSQ'):
            for stack in self.filepaths:
                self.readers.append(
                    CCDCBinaryReader(stack, self.fformat, self.datatype, 
                                     (self.y_size, self.x_size), self.n_band))


### Additional methods dealing with caching
    def cache_name_lookup(self, x, y):
        """ Return cache filename for given x/y """
        cache = 'n{n}_x{x}-y{y}_timeseries.npy'.format(
            n=self.length, x=x, y=y)
        if self.cache_folder is not None:
            return os.path.join(self.location, self.cache_folder, cache)
        else:
            return None


class CCDCBinaryReader(object):
    """
    This class defines the methods for reading pixel values from a raster
    dataset. I've coded this up because certain file formats are more
    efficiently accessed via fopen than via GDAL (i.e. BIP).

    http://osdir.com/ml/gdal-development-gis-osgeo/2007-04/msg00345.html

    Args:
    filename                    filename of the raster to read from
    fformat                     file format of the raster
    dt                          numpy datatype
    size                        list of [nrow, ncol]
    n_band                      number of bands in image
    """
    def __init__(self, filename, fformat, dt, size, n_band):
        self.filename = filename
        self.fformat = fformat
        self.dt = dt
        self.size = size
        self.n_band = n_band

        # Switch the actual definition of get_pixel by fformat
        # TODO: reimplement this using class inheritance
        # https://www.youtube.com/watch?v=miGolgp9xq8
        if fformat == 'BIP':
            self.get_pixel = self.__BIP_get_pixel

    def __BIP_get_pixel(self, row, col):
        if row < 0 or row >= self.size[0] or col < 0 or col >= self.size[1]:
            raise ValueError, 'Cannot select row,col %s,%s' % (row, col)

        with open(self.filename, 'rb') as f:
            # Skip to location of data in file
            f.seek(self.dt.itemsize * (row * self.size[1] + col) *
                self.n_band)
            # Read in
            dat = np.fromfile(f, dtype=self.dt, count=self.n_band)
            f.close()
            return dat
