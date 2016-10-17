import numpy as np
import pytest


@pytest.fixture(scope='function')
def prng():
    return np.random.RandomState(123456789)
