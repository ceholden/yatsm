""" Results storage in HDF5 datasets using PyTables
"""
import numpy as np
import tables as tb

SEGMENT_ATTRS = ['px', 'py', 'start', 'end', 'break']


def dtype_to_table(dtype):
    """ Convert a NumPy dtype to a PyTables Table description

    Essentially just :ref:`tables.descr_from_dtype` but it works on
    :ref:`np.datetime64`

    Args:
        dtype (np.dtype): NumPy data type

    Returns:
        dict: PyTables description
    """
    desc = {}

    for idx, name in enumerate(dtype.names):
        dt, _ = dtype.fields[name]
        if issubclass(dt.type, np.datetime64):
            tb_dtype = tb.Description({name: tb.Time64Col(pos=idx)})
        else:
            tb_dtype, byteorder = tb.descr_from_dtype(np.dtype([(name, dt)]))
        _tb_dtype = tb_dtype._v_colobjects
        _tb_dtype[name]._v_pos = idx
        desc.update(_tb_dtype)
    return desc


def create_table(h5file, result, index=True):
    """ Create table to store results

    Args:
        h5file (tables.file.File): PyTables HDF5 file
        result (OrderedDict[str, np.ndarray]): Results as a NumPy
            structured/record array, labeled according to the result type
        index (bool): Create index on :ref:`SEGMENT_ATTRS`

    Returns:
        table.table.Table:
    """
    first = list(result.keys())[0]
    for idx, (name, desc) in enumerate(result.items()):
        if idx == 0:
            table_desc = dtype_to_table(desc.dtype)
        else:
            table_desc.update({name: dtype_to_table(desc.dtype)})

    table = h5file.create_table(h5file.root, first, table_desc)

    if index:
        for attr in SEGMENT_ATTRS:
            getattr(table.col, attr).create_index()

    return table


class HDF5ResultsStore(object):
    """ PyTables based HDF5 results storage
    """
    pass
