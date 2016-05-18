""" Tools related to reading time series data using GDAL / rasterio
"""
from functools import partial
import os

import pandas as pd


def parse_dataset_file(input_file, date_format):
    """ Return parsed dataset CSV file as pd.DataFrame

    Args:
        input_file (str): CSV filename
        date_format (str): Format of date in input file

    Returns:
        pd.DataFrame: Dataset information
    """
    dt_parser = lambda x: pd.datetime.strptime(x, date_format)
    df = pd.read_csv(input_file,
                     parse_dates=['date'], date_parser=dt_parser)
    if not os.path.isabs(df['filename'][0]):
        _root = os.path.abspath(os.path.dirname(input_file))
        _root_join = partial(os.path.join, _root)
        df['filename'] = map(_root_join, df['filename'])

    return df


def get_coordinates(ul, res, window):
    """ Return Y/X coordinates of a raster to pass to xarray

    Args:
        ul (sequence[float, float]): Upper left Y/X coordinates
        res (sequence[float, float]): Pixel resolution Y/X
        window (): Number of
    """
    pass
