import contextlib
import os
import tempfile

import numpy as np
from osgeo import gdal, gdal_array
import pytest
import six
import yaml

gdal.AllRegister()
gdal.UseExceptions()


# CONFIG FILE UTILS
def deep_update(orig, upd):
    """ "Deep" modify all contents in ``origin`` with values in ``upd``
    """
    for k, v in six.iteritems(upd):
        if isinstance(v, dict):
            # recursively update sub-dictionaries
            _d = deep_update(orig.get(k, {}), v)
            orig[k] = _d
        else:
            orig[k] = v
    return orig


@pytest.fixture(scope='function')
def modify_config(request):
    @contextlib.contextmanager
    def _modify_config(f, d):
        """ Overwrites yaml file ``f`` with values in ``dict`` ``d`` """
        orig = yaml.load(open(f, 'r'))
        modified = orig.copy()
        try:
            modified = deep_update(modified, d)
            tmpcfg = tempfile.mkstemp(prefix='yatsm_', suffix='.yaml')[1]
            yaml.dump(modified, open(tmpcfg, 'w'))
            yield tmpcfg
        except:
            raise
        finally:
            os.remove(tmpcfg)
    return _modify_config


# RASTER READING UTILS
@pytest.fixture(scope='session')
def read_image():
    """ Read image ``f`` and return ``np.array`` of image values

    Image will be (nband, nrow, ncol)
    """
    def _read_image(f):
        ds = gdal.Open(f, gdal.GA_ReadOnly)
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(
            ds.GetRasterBand(1).DataType)
        nrow, ncol, nband = ds.RasterYSize, ds.RasterYSize, ds.RasterCount
        img = np.empty((nband, nrow, ncol), dtype=dtype)
        for i_b in range(nband):
            b = ds.GetRasterBand(i_b + 1)
            img[i_b, ...] = b.ReadAsArray()
        return img
    return _read_image
