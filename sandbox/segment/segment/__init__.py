""" Functions useful for utilizing spatial segments in timeseries
"""
import numpy as np


def segments_to_lines(segment, region_IDs):
    """ Returns the lines within a `segment` image containing `segment_ids`

    Args:
      segment (np.ndarray): 2D segmentation image
      region_IDs (iterable): sequence of region IDs within `segment` image

    Returns:
      np.ndarray: set of lines required to cover the specified region IDs

    """
    return np.unique(
        np.where(np.in1d(segment, region_IDs).reshape(segment.shape))[0])


