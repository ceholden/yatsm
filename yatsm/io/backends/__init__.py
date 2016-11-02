""" Time series reader backends
"""
from ._gdal import GDALTimeSeries

__all__ = [
    'GDALTimeSeries'
]

READERS = {
    'GDAL': GDALTimeSeries
}
