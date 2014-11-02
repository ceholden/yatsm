import numpy as np
import scipy.ndimage


def spatial_crossvalidate(roi, mask_values=[0], n_folds=3):
    """ Return crossvalidated accuracy for spatial training data

    Training data samples physically located next to test samples are likely to
    be strongly related due to spatial autocorrelation. This violation of
    independence will artificially inflate crossvalidated measures of
    algorithm performance. This function will performance crossvalidation by
    treating contiguous training data samples as one sample during
    crossvalidation.

    Args:
      roi (np.ndarray): training data raster to sample from
      mask_values (np.ndarray, optional): mask values to ignore from raster
      n_folds (int, optional): number of folds to use

    Returns:

    """
    if isinstance(mask_values, int):
        mask_values = [mask_values]

    mask = np.logical_or.reduce([roi == mv for mv in mask_values])

    labeled, n_labels = scipy.ndimage.label(roi)

    labels = np.unique(labeled)
    unmasked_labels = np.in1d(labels, np.unique(labeled[~mask]))
    labels = labels[unmasked_labels]

