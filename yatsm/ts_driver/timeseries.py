# -*- coding: utf-8 -*-
# vim: set expandtab:ts=4
"""
/***************************************************************************
 timeseries
                                 A QGIS plugin
 Plotting & visualization tools for time series analysis
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

import abc
import datetime as dt
import os

import numpy as np
import scipy.io

class AbstractTimeSeries(object):
    """ Abstract base class representing a remote sensing time series. 

    AbstractTimeSeries class is meant to be sub-classed and its methods 
    overriden. This interface simply defines attributes and methods expected 
    by "TSTools" QGIS plugin.
    
    Required attributes:
        image_names                 Names or IDs for each image
        filenames                   File basename for each image
        filepaths                   Full path to each image
        length                      Number of images in time series
        dates                       np.array of datetime for each image
        n_band                      number of bands per image
        x_size                      number of columns per image
        y_size                      number of rows per image
        geo_transform               geo-transform of images
        projection                  projection of images
        px                          current pixel column
        py                          current pixel row
        has_results                 boolean indicating existence of model fit

    Required methods:
        fetch_pixel                 retrieve pixel data for given x/y
        fetch_result                retrieve result for given x/y
        get_data                    return dataset
        get_prediction              return predicted dataset for x/y
        get_breaks                  return break points for time segments
    
    Additional attributes:
        has_cache                   boolean indicating existence of cached data
        can_cache                   boolean indicating potential to cache data
        cache_folder                location of cache, if any
        mask_band                   band (index on 0) of mask within images
        mask_val                    values to mask
    
    Additional methods:
        apply_mask                  apply mask to dataset
        retrieve_from_cache         retrieve dataset from cached retrieval
        write_to_cache              write retrieved dataset to cache

    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, location, image_pattern, stack_pattern):
        # Basic, required information
        self.location = os.path.realpath(location)
        self.image_pattern = image_pattern
        self.stack_pattern = stack_pattern

        # Overide/set these within subclasser as needed
        self.has_cache = False
        self.can_cache = False
        self.cache_folder = None
        self.mask_band = None
        self.mask_val = None

    def __repr__(self):
        return 'A {c} time series of {n} images at {m}'.format(
            c=self.__class__.__name__, n=self.length, m=hex(id(self)))

# ADDITIONAL METHODS: override/set by subclasser as needed
    def apply_mask(self, mask_band=None, mask_val=None):
        """ Use subclasser to set if capability is available """
        pass

    def retrieve_from_cache(self, x, y):
        """ Use subclasser to set if capability is available """
        return False

    def write_to_cache(self):
        """ Use subclasser to set if capability is available """
        return False

# REQUIRED PROPERTIES
    @abc.abstractproperty
    def image_names(self):
        """ Common names or IDs for each image """
        pass

    @abc.abstractproperty
    def filenames(self):
        """ File basename for each image """
        pass

    @abc.abstractproperty
    def filepaths(self):
        """ Full path to each image """
        pass

    @abc.abstractproperty
    def length(self):
        """ Length of the time series """
        pass 

    @abc.abstractproperty    
    def dates(self):
        """ np.array of datetime for each image """
        pass

    @abc.abstractproperty
    def n_band(self):
        """ number of bands per image """
        pass

    @abc.abstractproperty
    def x_size(self):
        """ number of columns per image """
        pass

    @abc.abstractproperty
    def y_size(self):
        """ number of rows per image """
        pass

    @abc.abstractproperty
    def geo_transform(self):
        """ geo-transform for each image """
        pass

    @abc.abstractproperty
    def projection(self):
        """ projection for each image """
        pass

    @abc.abstractproperty
    def has_results(self):
        """ boolean indicating existence of model fit """
        pass


# HELPER METHOD
    def get_ts_pixel(self, x, y):
        """ Fetch pixel data for a given x/y and set to self.data 
        
        Args:
            x                       column
            y                       row
                 
        """
        for i in xrange(self.length):
            self.retrieve_pixel(x, y, i)

# REQUIRED METHODS
    @abc.abstractmethod
    def retrieve_pixel(self, x, y, index):
        """ Return pixel data for a given x/y and index in time series

        Args:
            x                       column
            y                       row
            index                   index of image in time series

        Returns:
            data                    n_band x 1 np.array

        """
        pass

    @abc.abstractmethod
    def retrieve_result(self, x, y):
        """ Retrieve algorithm result for a given x/y

        Args:
            x                       column
            y                       row

        """
        pass

    @abc.abstractmethod
    def get_data(self, mask=True):
        """
        """
        pass

    @abc.abstractmethod
    def get_prediction(self, band):
        """
        """
        pass   

    @abc.abstractmethod
    def get_breaks(self, x, y):
        """
        """
        pass

    @abc.abstractmethod
    def get_px(self):
        """ current pixel column number """
        pass
    
    @abc.abstractmethod
    def set_px(self, value):
        """ set current pixel column number """
        pass

    @abc.abstractmethod
    def get_py(self):
        """ current pixel row number """
        pass

    @abc.abstractmethod    
    def set_py(self, value):
        """ set current pixel row number """
        pass

    _px = abc.abstractproperty(get_px, set_px)
    _py = abc.abstractproperty(get_py, set_py)


# Utility functions
def mat2dict(matlabobj):
    """
    Utility function:
    Converts a scipy.io.matlab.mio5_params.mat_struct to a dictionary
    """
    d = {}
    for field in matlabobj._fieldnames:
        value = matlabobj.__dict__[field]
        if isinstance(value, scipy.io.matlab.mio5_params.mat_struct):
            d[field] = mat2dict(value)
        else:
            d[field] = value
    return d

def ml2pydate(ml_date):
    """
    Utility function:
    Returns Python datetime for MATLAB date
    """
    return dt.datetime.fromordinal(int(ml_date)) - dt.timedelta(days = 366)

def py2mldate(py_date):
    """
    Utility function:
    Returns MATLAB datenum for Python datetime
    """
    return (py_date + dt.timedelta(days = 366)).toordinal()
