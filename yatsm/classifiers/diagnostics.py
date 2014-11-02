import logging

import numpy as np
import scipy.ndimage

from sklearn.utils import check_random_state
# from sklearn.cross_validation import KFold, StratifiedKFold

logger = logging.getLogger('yatsm')


class KFoldROI(object):
    """ Spatial cross validation iterator

    Training data samples physically located next to test samples are likely to
    be strongly related due to spatial autocorrelation. This violation of
    independence will artificially inflate crossvalidated measures of
    algorithm performance.

    Provides training and testing indices to split data into training and
    testing sets. Splits a "Region of Interest" image into k consecutive
    folds. Each fold is used as a validation set once while k - 1 remaining
    folds form the training set.

    Parameters:
      roi (np.ndarray): "Region of interest" matrix providing training data
        samples of some class

      n_folds (int, optional): Number of folds (default: 3)

      mask_values (int, list, tuple, or np.ndarray, optional): one or more
        values within roi to ignore from sampling (default: [0])

      shuffle (bool, optional): Shuffle the unique training data regions before
        splitting into batches (default: False)

      random_state (None, int, or np.random.RandomState): Pseudo-random number
        generator to use for random sampling. If None, default to numpy RNG
        for shuffling

    """

    shuffle = False

    def __init__(self, roi, n_folds=3, mask_values=[0], shuffle=False,
                 random_state=None):
        self.roi = roi
        self.n_folds = n_folds
        if isinstance(mask_values, (float, int)):
            self.mask_values = np.array([mask_values])
        elif isinstance(mask_values, (list, tuple)):
            self.mask_values = np.array(mask_values)
        elif isinstance(mask_values, np.ndarray):
            self.mask_values = mask_values
        else:
            raise TypeError('mask_values must be float, int, list, tuple,'
                            ' or np.ndarray')
        if shuffle:
            self.shuffle = True
            self.rng = check_random_state(random_state)

        self._label_roi()

    def __iter__(self):
        n = self.n
        n_folds = self.n_folds

        fold_sizes = (n // n_folds) * np.ones(n_folds, dtype=np.int)
        fold_sizes[:n % n_folds] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            test_i = np.in1d(self.indices[:, 0], self.labels[start:stop])
            train_i = np.in1d(self.indices[:, 0], self.labels[stop:])

            yield ((self.indices[test_i, 1], self.indices[test_i, 2]),
                   (self.indices[train_i, 1], self.indices[train_i, 2]))
            current = stop

    def _label_roi(self):
        """ Internal method to label region of interest image
        """
        labeled, n = scipy.ndimage.label(self.roi)

        labels = np.unique(labeled)
        self.labels = labels[~np.in1d(labels, self.mask_values)]
        self.n = self.labels.size

        n_samples = (~np.in1d(self.roi, self.mask_values)).sum()
        self.indices = np.zeros((n_samples, 3), dtype=np.int)
        _start = 0

        for l in self.labels:
            _n = (labeled == l).sum()
            _row, _col = np.where(labeled == l)
            self.indices[_start:_start + _n, 0] = l
            self.indices[_start:_start + _n, 1] = _row
            self.indices[_start:_start + _n, 2] = _col
            _start += _n

        if self.shuffle:
            self.rng.shuffle(self.labels)
