""" Time series reader backends
"""
from ._gdal import GDALTimeSeries


READERS = {
    'GDAL': GDALTimeSeries
}
