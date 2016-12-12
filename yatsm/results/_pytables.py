""" Results storage in HDF5 datasets using PyTables
"""
import errno
import os

import numpy as np
import tables as tb

SEGMENT_ATTRS = ['px', 'py', 'start', 'end', 'break']

FILTERS = tb.Filters(complevel=1, complib='zlib', shuffle=True)


def _has_node(h5, node, **kwargs):
    try:
        h5.get_node(node, **kwargs)
    except tb.NoSuchNodeError:
        return False
    else:
        return True


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


def create_table(h5file, where, name, result, index=True,
                 expectedrows=10000, **table_config):
    """ Create table to store results

    Args:
        h5file (tables.file.File): PyTables HDF5 file
        where (str or tables.group.Group): Parent group to place table
        name (str): Name of new table
        result (np.ndarray): Results as a NumPy structured array
        index (bool): Create index on :ref:`SEGMENT_ATTRS`
        expectedrows (int): Expected number of rows to store in table
        table_config (dict): Additional keyword arguments to be passed
            to ``h5file.create_table``

    Returns:
        table.table.Table: HDF5 table
    """
    table_desc = dtype_to_table(result.dtype)

    if _has_node(h5file, where, name=name):
        table = h5file.get_node(where, name=name)
    else:
        table = h5file.create_table(where, name,
                                    description=table_desc,
                                    expectedrows=expectedrows,
                                    createparents=True,
                                    **table_config)
        if index:
            for attr in SEGMENT_ATTRS:
                getattr(table.cols, attr).create_index()

    return table


def create_task_groups(h5file, tasks, filters=FILTERS, overwrite=False):
    """ Create groups for tasks

    Args:
        h5file (tables.file.File): PyTables HDF5 file
        tasks (list[Task]): A list of ``Task`` to create nodes
            from
        filters (table.Filter): PyTables filter
        overwrite (bool): Allow overwriting of existing table

    Returns:
        tuple (Task, tables.group.Group): Each task that creates a group and
        the group it created
    """
    groups = []
    for task in tasks:
        if task.output_record:
            where, name = task.record_result_group(tasks)
            if not _has_node(h5file, where, name=name) or overwrite:
                # TODO: check w/ w/o createparents -- shouldn't need it
                g = h5file.create_group(where, name, title=task.name,
                                        filters=filters, createparents=False)
            else:
                g = h5file.get_node(where, name)
            groups.append((task, g))

    return groups


def create_task_tables(h5file, tasks, results, filters=FILTERS,
                       overwrite=False, **tb_config):
    """ Create groups for tasks

    Args:
        h5file (tables.file.File): PyTables HDF5 file
        tasks (list[Task]): A list of ``Task`` to create nodes
            from
        results (dict): Result :ref:`np.ndarray` structure arrays organized by
            name in a dict
        filters (table.Filter): PyTables filter
        overwrite (bool): Allow overwriting of existing table

    Returns:
        list[tuple (Task, tables.Table)]: Each task that creates a group and
        the group it created
    """
    tables = []
    for task in tasks:
        if task.output_record:
            where, name = task.record_result_group(tasks)
            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            t = create_table(h5file, where, name,
                             results[task.output_record],
                             index=True,
                             **tb_config)
            tables.append((task, t))
    return tables


class HDF5ResultsStore(object):
    """ PyTables based HDF5 results storage
    """
    pass
