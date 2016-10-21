from collections import namedtuple

import pandas as pd
import xarray as xr

pandas_like = (pd.DataFrame, pd.Series, xr.DataArray)

_fields = ['method', 'index', 'score', 'process', 'boundary', 'pvalue', 'signif']

#: namedtuple: Structural break detection results
StructuralBreakResult = namedtuple('StructuralBreakResult', _fields)
