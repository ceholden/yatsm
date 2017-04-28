from functools import partial
import os
from pathlib import Path
from tempfile import mkdtemp

if os.environ.get('TRAVIS'):
    # use agg backend on TRAVIS for testing so DISPLAY isn't required
    import matplotlib as mpl
    mpl.use('agg')


import numpy as np  # noqa
import pandas as pd  # noqa
import pytest  # noqa
import yaml  # noqa

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
