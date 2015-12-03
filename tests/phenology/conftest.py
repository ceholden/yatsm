import os

import numpy as np
import pandas as pd
import pytest

here = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def df(request):
    """ Return a Pandas df of example EVI timeseries """
    df = pd.read_csv(os.path.join(here, 'data', 'evi_test.csv'))
    df.columns = ['yr', 'doy', 'prd', 'evi']

    # Mask invalid EVI data
    mask = np.where((df['evi'] >= 0) & (df['evi'] <= 1))[0]
    df = df.ix[mask]

    return df
