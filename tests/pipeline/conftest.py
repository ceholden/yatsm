""" Examples as fixtures for pipeline testing
"""
import pytest


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


@pytest.fixture
def tasks1(config1):
    return config1['pipeline']['tasks']


@pytest.fixture
def tasks2(config2):
    return config2['pipeline']['tasks']


@pytest.fixture
def tasks3(config3):
    return config3['pipeline']['tasks']
