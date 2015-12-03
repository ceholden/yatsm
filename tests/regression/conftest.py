import os

import pandas as pd
import pytest

here = os.path.dirname(__file__)


@pytest.fixture(scope='function')
def airquality(request):
    airquality = pd.read_csv(os.path.join(here, 'data', 'airquality.csv'))
    airquality.columns = ['Unnamed', 'Ozone', 'SolarR', 'Wind',
                          'Temp', 'Month', 'Day']
    airquality = airquality.dropna()

    return airquality
