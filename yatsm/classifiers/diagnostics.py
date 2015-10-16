import logging

import numpy as np
import scipy.ndimage

from sklearn.utils import check_random_state
# from sklearn.cross_validation import KFold, StratifiedKFold

logger = logging.getLogger('yatsm')


def kfold_scores(X, y, algo, kf_generator):
    """ Performs KFold crossvalidation and reports mean/std of scores

    Args:
      X (np.ndarray): X feature input used in classification
      y (np.ndarray): y labeled examples
      algo (sklean classifier): classifier used from scikit-learn
      kf_generator (sklearn crossvalidation generator): generator for indices
        used in crossvalidation

    Returns:
      (mean, std): mean and standard deviation of crossvalidation scores

    """
    scores = np.zeros(kf_generator.n_folds)
    for i, (train, test) in enumerate(kf_generator):
        scores[i] = algo.fit(X[train, :], y[train]).score(X[test, :], y[test])

    logger.info('scores: {0}'.format(scores))
    logger.info('score mean/std: {0}/{1}'.format(scores.mean(), scores.std()))

    return scores.mean(), scores.std()


class SpatialKFold(object):
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
      y (np.ndarray): Labeled features

      row (np.ndarray): Row (y) pixel location for each `y`

      col (np.ndarray): Column (x) pixel location for each `x`

      n_folds (int, optional): Number of folds (default: 3)

      shuffle (bool, optional): Shuffle the unique training data regions before
        splitting into batches (default: False)

      random_state (None, int, or np.random.RandomState): Pseudo-random number
        generator to use for random sampling. If None, default to numpy RNG
        for shuffling

    """

    shuffle = False

    def __init__(self, y, row, col, n_folds=3, shuffle=False,
                 random_state=None):
        if y.size != row.size or y.size != col.size:
            raise ValueError('Labels provided (y) must be the same size as '
                             'the row and columns provided')
        self.y = y
        self.row = row
        self.col = col
        self.n_folds = n_folds

        if shuffle:
            self.shuffle = True
            self.rng = check_random_state(random_state)

        self._recreate_labels()

    def __iter__(self):
        fold_sizes = (self.n // self.n_folds) * np.ones(self.n_folds,
                                                        dtype=np.int)
        fold_sizes[:self.n % self.n_folds] += 1
        current = 0

        ind = np.arange(self.y.size)

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size

            test_i = self._labels_to_indices(self.labels[start:stop])

            yield ind[test_i], ind[~test_i]
            current = stop

    def _recreate_labels(self):
        """ Internal method to label regions of `self.y` from pixel locations
        """
        roi = np.zeros((self.row.max() + 1, self.col.max() + 1),
                       dtype=self.y.dtype)
        roi[self.row, self.col] = self.y

        self.labeled, _ = scipy.ndimage.label(roi)
        self.labels = np.unique(self.labeled[self.labeled != 0])
        self.n = self.labels.size

        if self.shuffle:
            self.rng.shuffle(self.labels)

        self.indices = []

    def _labels_to_indices(self, labels):
        lab_row, lab_col = np.where(np.in1d(
            self.labeled, labels).reshape(self.labeled.shape))
        return np.logical_and(np.in1d(self.row, lab_row),
                              np.in1d(self.col, lab_col))


class SpatialKFold_ROI(object):
    """ Spatial cross validation iterator on ROI images

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
