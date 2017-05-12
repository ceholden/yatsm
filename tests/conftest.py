import os
from pathlib import Path
from tempfile import mkdtemp

import yaml  # noqa

if os.environ.get('TRAVIS'):
    # use agg backend on TRAVIS for testing so DISPLAY isn't required
    import matplotlib as mpl
    mpl.use('agg')

# We wait to import anything that might deal with matplotlib until
# after we've set the backend...
import numpy as np  # noqa
import pandas as pd  # noqa
import pytest  # noqa
import xarray as xr  # noqa

HERE = Path(__file__).resolve().parent
DATA = HERE.joinpath('data')


# DATASETS
@pytest.fixture(scope='function')
def airquality(request):
    airquality = pd.read_csv(str(DATA.joinpath('airquality.csv')))
    airquality.columns = ['Unnamed', 'Ozone', 'SolarR', 'Wind',
                          'Temp', 'Month', 'Day']
    airquality = airquality.dropna()

    return airquality


@pytest.fixture(scope='module')
def HRF1_filename():
    return str(DATA.joinpath('HRF1_Block_0-10_0-10.nc'))


@pytest.fixture(scope='module')
def HRF1_data_vars():
    return ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'temp', 'fmask']


@pytest.fixture(scope='module')
def HRF1_ds(HRF1_filename):
    """ Harvard Forest Research tower footprint dataset (unmasked/unscaled)

    Tip: Use `xr.decode_cf` to apply the masking and scaling
    """
    return xr.open_dataset(HRF1_filename, mask_and_scale=False)


@pytest.fixture(scope='module')
def HRF1_da(HRF1_ds, HRF1_data_vars):
    """ xr.DataArray of just the `HRF1_data_vars`
    """
    return HRF1_ds[HRF1_data_vars].to_array('band')


# PIPELINES
@pytest.fixture(scope='module')
def configfile1():
    return str(DATA.joinpath('config1.yaml'))


@pytest.fixture(scope='module')
def configfile2():
    return str(DATA.joinpath('config2.yaml'))


@pytest.fixture(scope='module')
def configfile3():
    return str(DATA.joinpath('config3.yaml'))


@pytest.fixture(scope='module')
def config1(configfile1):
    """ A long(er) example of a configuration
    """
    return yaml.load(open(configfile1))


@pytest.fixture(scope='module')
def config2(configfile2):
    """ Shorter, but okay configuration
    """
    return yaml.load(open(configfile2))


@pytest.fixture(scope='module')
def config3(configfile3):
    """ This configuration has unmet dependencies
    """
    return yaml.load(open(configfile3))


# MISC
@pytest.fixture(scope='function')
def mkdir_permissions(request):
    """ Fixture for creating dir with specific read/write permissions """
    def make_mkdir(read=False, write=False):
        if read and write:
            mode = 0o755
        elif read and not write:
            mode = 0o555
        elif not read and write:
            mode = 0o333
        elif not read and not write:
            mode = 0o000

        path = mkdtemp()
        os.chmod(path, mode)

        def fin():
            os.chmod(path, 0o755)
            os.removedirs(path)
        request.addfinalizer(fin)

        return path

    return make_mkdir
