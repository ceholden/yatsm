""" Custom transforms for Patsy formulas

TODO:
    - Add vegetation indices

"""
import numpy as np
from patsy import stateful_transform


class Harmonic(object):
    """ Transform `x` into seasonal harmonics with a given frequency

    Args:
      x (np.ndarray): array of ordinal dates
      freq (int or float): frequency of harmonics

    """
    def __init__(self):
        self.w = 2 * np.pi / 365.25

    def memorize_chunk(self, x, freq):
        pass

    def memorize_finish(self):
        pass

    def transform(self, x, freq):
        x = np.asarray(x)
        if x.shape == ():
            x = x[np.newaxis]

        return np.array([
            np.cos(freq * self.w * x),
            np.sin(freq * self.w * x)
        ]).T


harm = stateful_transform(Harmonic)
