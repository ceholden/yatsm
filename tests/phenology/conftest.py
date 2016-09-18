from os.path import dirname, join

import pandas as pd
import pytest


@pytest.fixture(scope='module')
def data():
    data = join(dirname(__file__), 'data', 'evi_test.csv')
    df = pd.read_csv(data, parse_dates=['time'], index_col='time')

    mask = ((df['evi'] >= 0) & (df['evi'] <= 1))

    return df.loc[mask]
