""" Examples as fixtures for pipeline testing
"""
import os

import pytest
import yaml

HERE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(HERE, 'data')


@pytest.fixture(scope='module')
def config_1(request):
    """ A long(er) example of a configuration
    """
    return yaml.load(open(os.path.join(DATA, 'config1.yaml')))


@pytest.fixture(scope='module')
def config_2(request):
    """ Shorter, but okay configuration
    """
    return yaml.load(open(os.path.join(DATA, 'config2.yaml')))


@pytest.fixture(scope='module')
def config_3(request):
    """ This configuration has unmet dependencies
    """
    return yaml.load(open(os.path.join(DATA, 'config3.yaml')))


@pytest.fixture
def pipe_defn(request):
    return {
        'data': {  # in practice, this would be an xarray
            'red': 5,
            'nir': 60,
            'swir1': 25,
            'swir2': 15
        },
        'record': {}
    }
