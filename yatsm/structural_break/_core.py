from collections import namedtuple

import pandas as pd
import xarray as xr

pandas_like = (pd.DataFrame, pd.Series, xr.DataArray)

_fields = ['method', 'index', 'score', 'process', 'pvalue', 'signif']

#: namedtuple: Structural break detection results
StructuralBreakResult = namedtuple('StructuralBreakResult', _fields)
