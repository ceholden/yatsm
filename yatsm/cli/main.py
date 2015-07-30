""" Loads all commands for YATSM command line interface

Modeled after very nice `click` interface for `rasterio`:
https://github.com/mapbox/rasterio/blob/master/rasterio/rio/main.py

"""
from pkg_resources import iter_entry_points

from yatsm.cli.cli import cli


for entry_point in iter_entry_points('yatsm.yatsm_commands'):
    try:
        entry_point.load()
    except:
        from IPython.core.debugger import Pdb
        Pdb().set_trace()
