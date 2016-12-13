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
            where, tablename = task.record_result_location(tasks)
            group, groupname = s.rsplit('/', 1)
            if not _has_node(h5file, where, name=name):
                g = h5file.create_group(group, groupname, title=task.name,
                                        filters=filters, createparents=False)
            else:
                g = h5file.get_node(group, groupname)
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
            where, tablename = task.record_result_location(tasks)
            # from IPython.core.debugger import Pdb; Pdb().set_trace()
            t = create_table(h5file, where, tablename,
                             results[task.output_record],
                             index=True,
                             **tb_config)
            tables.append((task, t))
    return tables


class HDF5ResultsStore(object):
    """ PyTables based HDF5 results storage

    Args:
        filename (str): HDF5 file
        mode (str): File mode to open with
        keep_open (bool): Keep file handle open after calls
        tb_kwargs: Optional keywork arguments to :ref:`tables.open_file`
    """
    def __init__(self, filename, mode=None, keep_open=True,
                 **tb_kwargs):
        self.filename = filename
        self.mode = mode or ('r+' if os.path.exists(self.filename) else 'w')
        self.keep_open = keep_open
        self.tb_kwargs = tb_kwargs
        self.h5file = None

# WRITING
    @staticmethod
    def _write_row(h5file, result, tables):
        for task, table in tables:
            table.append(result[task.output_record])
            table.flush()

    def write_result(self, pipeline, result, overwrite=True):
        """ Write result to HDF5

        Args:
            result (dict): Dictionary of pipeline 'record' results
                where key is task output and value is a structured
                :ref:`np.ndarray`
            overwrite (bool): Overwrite existing values

        Returns:
            HDF5ResultsStore

        """
        result = result.get('record', result)
        with self as store:
            tasks = list(pipeline.tasks.values())
            tables = create_task_tables(self.h5file, tasks, result,
                                        overwrite=True)
            store._write_row(store.h5file, result, tables)

        return self


# CONTEXT HELPERS
    def __enter__(self):
        if isinstance(self.h5file, tb.file.File):
            if (getattr(self.h5file, 'mode', '') == self.mode
                    and self.h5file.isopen):
                return self  # already opened in correct form, bail
            else:
                self.h5file.close()
        else:
            try:
                os.makedirs(os.path.dirname(self.filename))
            except OSError as er:
                if er.errno == errno.EEXIST:
                    pass
                else:
                    raise

        self.h5file = tb.open_file(self.filename, mode=self.mode, title='YATSM',
                                   **self.tb_kwargs)

        return self

    def __exit__(self, *args):
        if self.h5file and not self.keep_open:
            self.h5file.close()

    def __del__(self):
        self.h5file.close()

    def close(self):
        if self.h5file:
            self.h5file.close()

# DICT LIKE
    def keys(self):
        """ Yields HDF5 file nodes names
        """
        with self as store:
            for node in store.h5file.walk_nodes():
                yield node._v_pathname

    def items(self):
        """ Yields key/value pairs for groups
        """
        with self as store:
            for group in store.h5file.walk_groups():
                yield group._v_pathname, group

# TODO: add getitem/setitem and getattr
# TODO: add __repr__
