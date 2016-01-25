""" Helper classes for reading individual lines from stacked timeseries images

The classes, and instances of these classes, are designed to decrease the
number of relatively expensive file open calls that are made unnecessarily
if reading from the same files over and over. When using GDAL, for example,
tests have shown that unnecessary opening and closing of images and image
bands is slower compared to keeping the file reference and opening the files
only once.

Attributes:

    bip_reader (_BIPStackReader): instance of :class:`_BIPStackReader` that
        reads from Band-Interleave-by-Pixel (BIP) images
    gdal_reader (_GDALStackReader): instance of :class:`_GDALStackReader` that
        reads from file formats supported by GDAL
"""
import numpy as np
from osgeo import gdal, gdal_array

gdal.AllRegister()
gdal.UseExceptions()


class _BIPStackReader(object):
    """ Simple class to read BIP formatted stacks

    Some tests have shown that we can speed up total dataset read time by
    storing the file object references to each image as we loop over many rows
    instead of opening once per row read. This is a simple class designed to
    store these references.

    Note that this class assumes the images are "stacked" -- that is that all
    images contain the same number of rows, columns, and bands, and the images
    are of the same geographic extent.

    Attributes:
        filenames (list): list of filenames to read from
        n_image (int): number of images
        size (tuple): tuple of (int, int) containing the number of columns and
            bands in the image
        datatype (np.dtype): NumPy datatype of images
        files (list): list of file objects for each image

    """
    filenames = []

    def _init_attrs(self, filenames):
        self.filenames = filenames
        self.files = [open(f, 'rb') for f in self.filenames]

        self.n_image = len(self.filenames)
        ds = gdal.Open(self.filenames[0], gdal.GA_ReadOnly)
        self.size = (ds.RasterXSize, ds.RasterCount)
        self.datatype = gdal_array.GDALTypeCodeToNumericTypeCode(
            ds.GetRasterBand(1).DataType)

    def _read_row(self, row):
        data = np.empty((self.size[1], self.n_image, self.size[0]),
                        self.datatype)

        for i, fid in enumerate(self.files):
            # Find where we need to seek to
            offset = np.dtype(self.datatype).itemsize * \
                (row * self.size[0]) * self.size[1]
            # Seek relative to current position
            fid.seek(offset - fid.tell(), 1)
            # Read
            data[:, i, :] = np.fromfile(fid,
                                        dtype=self.datatype,
                                        count=self.size[0] * self.size[1],
                                        ).reshape(self.size).T

        return data

    def read_row(self, filenames, row):
        """ Return a 3D NumPy array (nband x nimage x ncol) of one row's data

        Args:
            filenames (iterable): list of filenames to read from
            row (int): row in image to return

        Returns:
            np.ndarray: 3D NumPy array (nband x nimage x ncol) of image
                data for desired row

        """
        if not np.array_equal(filenames, self.filenames):
            self._init_attrs(filenames)
        return self._read_row(row)


class _GDALStackReader(object):
    """ Simple class to read stacks using GDAL, keeping file objects open

    Some tests have shown that we can speed up total dataset read time by
    storing the file object references to each image as we loop over many rows
    instead of opening once per row read. This is a simple class designed to
    store these references.

    Note that this class assumes the images are "stacked" -- that is that all
    images contain the same number of rows, columns, and bands, and the images
    are of the same geographic extent.

    Attributes:
        filenames (list): list of filenames to read from
        n_image (int): number of images
        n_band (int): number of bands in an image
        n_col (int): number of columns per row
        datatype (np.dtype): NumPy datatype of images
        datasets (list): list of GDAL datasets for all filenames
        dataset_bands (list): list of lists containing all GDAL raster band
            datasets, for all image filenames

    """
    filenames = []

    def _init_attrs(self, filenames):
        self.filenames = filenames
        self.datasets = [gdal.Open(f, gdal.GA_ReadOnly) for
                         f in self.filenames]

        self.n_image = len(filenames)
        self.n_band = self.datasets[0].RasterCount
        self.n_col = self.datasets[0].RasterXSize
        self.datatype = gdal_array.GDALTypeCodeToNumericTypeCode(
            self.datasets[0].GetRasterBand(1).DataType)

        self.dataset_bands = [
            [ds.GetRasterBand(i + 1) for i in range(self.n_band)]
            for ds in self.datasets
        ]

    def _read_row(self, row):
        data = np.empty((self.n_band, self.n_image, self.n_col),
                        self.datatype)
        for i, ds_bands in enumerate(self.dataset_bands):
            for n_b, band in enumerate(ds_bands):
                data[n_b, i, :] = band.ReadAsArray(0, row, self.n_col, 1)
        return data

    def read_row(self, filenames, row):
        """ Return a 3D NumPy array (nband x nimage x ncol) of one row's data

        Args:
            filenames (iterable): list of filenames to read from
            row (int): row in image to return

        Returns:
            np.ndarray: 3D NumPy array (nband x nimage x ncol) of image
                data for desired row

        """
        if not np.array_equal(filenames, self.filenames):
            self._init_attrs(filenames)
        return self._read_row(row)


bip_reader = _BIPStackReader()
gdal_reader = _GDALStackReader()
