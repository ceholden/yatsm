""" Geospatial utility functions
"""
import shapely.geometry


def bounds_to_polygon(bounds):
    """ Returns Shapely polygon of bounds

    Args:
        bounds (iterable): bounds (left bottom right top)

    Returns:
        shapely.geometry.Polygon: polygon of bounds

    """
    return shapely.geometry.Polygon([
        (bounds[0], bounds[1]),
        (bounds[2], bounds[1]),
        (bounds[2], bounds[3]),
        (bounds[0], bounds[3])
    ])
