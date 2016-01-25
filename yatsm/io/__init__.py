""" YATSM IO module

TODO: include result file IO abstraction (issue #69)

Contents:

    * ``helpers``: Collection of helper functions that ease common filesystem
      operations
    * ``stack_line_reader.py``: Two readers of stacked timeseries images that
      trade storing file handles for reducing repeated and relatively expensive
      file open calls
"""
from .helpers import find_stack_images, mkdir_p
from .readers import (get_image_attribute, read_image, read_pixel_timeseries,
                      read_line)
from .stack_line_readers import bip_reader, gdal_reader


__all__ = [
    'find_stack_images', 'mkdir_p',
    'bip_reader', 'gdal_reader',
    'get_image_attribute', 'read_image', 'read_pixel_timeseries', 'read_line'
]
