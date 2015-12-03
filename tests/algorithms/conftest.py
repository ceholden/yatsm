import os

import numpy as np
import pytest

here = os.path.dirname(__file__)


@pytest.fixture(scope='session')
def example_data(request):
    return np.load(os.path.join(here, 'data', 'example_data.npz'))
