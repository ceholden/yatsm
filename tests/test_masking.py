""" Tests for yatsm.masking
"""
import numpy as np
import pytest
import yatsm.masking


@pytest.fixture(scope='module')
def masking_data(request):
    # Two years, 8 day repeat
    x = np.arange(735851, 735851 + 365 * 2, 8)

    # Simulate some timeseries in green & swir1
    def seasonality(x, amp):
        return np.cos(2 * np.pi / 365.25 * x) * amp
    green = np.ones_like(x) * 1000 + seasonality(x, 750)
    swir1 = np.ones_like(x) * 1250 + seasonality(x, 500)
    Y = np.vstack((green, swir1))

    # Add in some noise
    idx_green_noise = 15
    idx_swir1_noise = 30

    Y[0, idx_green_noise] = 8000
    Y[1, idx_swir1_noise] = 10

    return x, Y, np.array([idx_green_noise, idx_swir1_noise])


def test_multitemp_mask(masking_data):
    x, Y, idx_noise = masking_data
    n_year = 2

    # Find mask
    mask = yatsm.masking.multitemp_mask(x, Y, n_year, green=0, swir1=1)

    assert np.array_equal(np.where(~mask)[0], idx_noise)


def test_smooth_mask(masking_data):
    x, Y, idx_noise = masking_data
    span = 16

    # Find mask
    mask = yatsm.masking.smooth_mask(x, Y, span, green=0, swir1=1)

    assert np.array_equal(np.where(~mask)[0], idx_noise)
