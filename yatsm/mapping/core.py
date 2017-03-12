""" Core mapping functions and utilities
"""
import logging

import numpy as np
import tables as tb
import rasterio.transform

logger = logging.getLogger(__name__)


def result_map(out, results, table, attr_columns,
               attr_funcs=None, **query_kwds):
    """ Populate `out` from data queried from saved record model results

    Args:
        out (np.ndarray): 2D or 3D (nband x nrow x ncol) array to fill result
            data into
        results (iterable): A list of :class:`HDF5ResultsStore` files
        table (str): The table to retrieve the data from
        attr_columns (tuple): Attributes from results table to map. The number
            of attributes given should be the same as the number of bands in
            `out`.
        attr_funcs (iterable): Optionally, provide a function to apply to each
            attribute described in `attr_columns`. Please supply `None`, or an
            iterable of either a `callable` object or `None` for each attribute
            in `attr_columns`.
        query_kwds (dict): Additional search terms to pass to
            :meth:`HDF5ResultsStore.query`

    Returns:
        np.ndarray: `out`, but with desired result file attributes mapped into
            the image
    """
    columns = ('px', 'py', ) + attr_columns

    def guard_out(out):
        shape = (1, ) * (3 - out.ndim) + out.shape
        return np.atleast_3d(out).reshape(*shape)

    out = guard_out(out)

    assert out.ndim == 3, '`guard_out` should have worked!'
    if out.shape[0] != len(attr_columns):
        raise ValueError(
            'Provided `out` must have "{0}" bands to store '
            '"{1!r}" but it has "{2}" number of bands'
            .format(len(attr_columns), attr_columns, out.shape[0]))

    for _result in results:
        try:
            with _result as result:
                if not table:
                    from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA
                segs = result.query(table, columns=columns, **query_kwds)
                y, x = rasterio.transform.rowcol(result.transform,
                                                 segs['px'],
                                                 segs['py'])
                for bidx, attr in enumerate(attr_columns):
                    out[bidx, y, x] = segs[attr]
        except tb.exceptions.HDF5ExtError as err:
            logger.error('Result file {} is corrupt or unreadable'
                         .format(_result.filename), err)

    return out
