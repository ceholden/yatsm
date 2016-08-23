""" Utilities for extracting training data from region of interests (ROI)
"""
import numpy as np
from rasterio.features import rasterize
from shapely.geometry import shape as geom_shape


def extract_roi(raster, vector, feature_prop=None, all_touched=False):
    """ Yield pixel data from ``src`` for ROIs in ``features``

    Args:
        raster (rasterio.RasterReader): The ``rasterio`` dataset used to
            extract training data values from
        vector (list[dict]): A list of features from a polygon vector file as
            GeoJSON-like
        feature_prop (str): The name of the attribute from ``features``
            containing the ROI labels
        all_touched (bool): Rasterization option that decides if all pixels
            touching the ROI should be included, or just pixels from within
            the ROI

    Returns:
        tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray): A tuple
            containing an array of ROI data from ``src`` (``band x n``), the
            ROI data label (``n``), and the X and Y coordinates of each data
            point (``n`` and ``n`` sized)

    """
    if not feature_prop:
        feature_prop = list(vector[0]['properties'].keys())[0]

    for feat in vector:
        geom = geom_shape(feat['geometry'])
        label = feat['properties'][feature_prop]
        bounds = tuple(geom.bounds)

        window = raster.window(*bounds, boundless=True)
        data = raster.read(window=window, boundless=True)
        shape = data.shape
        transform = raster.window_transform(window)

        roi = rasterize(
            [(feat['geometry'], 1)],
            out_shape=shape[1:],
            transform=transform,
            fill=0,
            all_touched=all_touched
        )

        mask = roi == 0
        if raster.nodata:
            mask = np.logical_or((data == raster.nodata).any(axis=0), mask)

        masked = np.ma.MaskedArray(
            data,
            mask=np.ones_like(data) * mask
        )

        ys, xs = np.where(~mask)
        coord_xs, coord_ys = transform * (xs, ys)

        masked = masked.compressed()
        npix = masked.size / shape[0]
        masked = masked.reshape((shape[0], npix))

        label = np.repeat(label, coord_ys.size)

        yield (masked, label, coord_xs, coord_ys, )
