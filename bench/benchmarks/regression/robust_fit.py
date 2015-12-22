""" Benchmark for ``yatsm.regresion.robust_fit``
"""
from yatsm.regression.robust_fit import RLM

from ..bench_utils.example_timeseries import PixelTimeseries


class BenchRLM(PixelTimeseries):
    """ Benchmark robust linear model ``RLM`` calculation
    """
    def time_RLM(self, setup):
        """ Time robust linear model for 7 series 500 times
        """
        for y in setup['Y'][:-1, :]:
            for i in range(500):
                RLM().fit(setup['X'], y)
