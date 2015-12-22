""" Benchmark for ``yatsm.regresion.recresid``
"""
from yatsm.regression.recresid import recresid

from ..bench_utils.example_timeseries import PixelTimeseries


class BenchRecresid(PixelTimeseries):
    """ Benchmark recursive residual calculation
    """
    def time_recresid(self, setup):
        """ Time recursive residuals for 7 series 500 times each
        """
        for y in setup['Y'][:-1, :]:
            for i in range(500):
                recresid(setup['X'], y)
