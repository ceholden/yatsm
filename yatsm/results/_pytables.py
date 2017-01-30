""" Results storage in HDF5 datasets using PyTables
"""
import errno
import logging
import os

import numpy as np
from rasterio.crs import CRS
from rasterio.coords import BoundingBox
import six
import shapely.wkt
import tables as tb

from yatsm.algorithms import SEGMENT_ATTRS
from yatsm.gis import bounds_to_polygon
from yatsm.results.utils import result_filename, RESULT_TEMPLATE

logger = logging.getLogger(__name__)

FILTERS = tb.Filters(complevel=1, complib='zlib', shuffle=True)


def _has_node(h5, node, **kwds):
    try:
        h5.get_node(node, **kwds)
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
        logger.debug('Returning existing table %s/%s' % (where, name))
        table = h5file.get_node(where, name=name)
    else:
        logger.debug('Creating new table %s/%s' % (where, name))
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
        tb_kwds: Optional keywork arguments to :ref:`tables.open_file`
    """
    def __init__(self, filename, crs=None, bounds=None,
                 mode=None, keep_open=True, **tb_kwds):
        _exists = os.path.exists(filename)
        if crs and not isinstance(crs, six.string_types):
            raise TypeError('`crs` must be given as `str`')
        if bounds and not isinstance(bounds, six.string_types):
            raise TypeError('`bounds` must be given as `str`')
        if not _exists and not crs and not bounds:
            raise TypeError('Must specify `crs` and `bounds` when creating '
                            'new file')
        self.filename = filename
        self._crs = crs
        self._bounds = bounds
        self.mode = mode or 'r+' if _exists else 'w'
        self.keep_open = keep_open
        self.tb_kwds = tb_kwds
        self.h5file = None

        logger.debug('Opening %s in mode %s' % (self.filename, self.mode))

# CREATION
    @classmethod
    def from_window(cls, window, crs=None, bounds=None, reader=None,
                    root='.', pattern=RESULT_TEMPLATE,
                    **open_kwds):
        """ Return instance of class for a given window

        When creating a file, specify the ``crs`` and ``bounds`` of the results
        you're about to store. You can either do this directly by specifying
        ``crs`` and ``bounds``, or indirectly by providing the ``reader``
        used as the grid point of reference
        """
        filename = result_filename(window, root=root, pattern=pattern)

        if reader:
            crs = reader.crs.to_string()
            bounds = bounds_to_polygon(reader.window_bounds(window)).wkt

        return cls(filename, crs=crs, bounds=bounds, **open_kwds)

# WRITING
    @staticmethod
    def _write_row(h5file, result, tables):
        for task, table in tables:
            table.append(result[task.output_record])
            table.flush()

    def write_result(self, pipeline, result, overwrite=True):
        """ Write result to HDF5

        Args:
            pipeline (yatsm.pipeline.Pipeline): YATSM pipeline of tasks
            result (dict): Dictionary of pipeline 'record' results
                where key is task output and value is a structured
                :ref:`np.ndarray`
            overwrite (bool): Overwrite existing values
            attrs (dict): KEY=VALUE items to put in to each table's
                ``attr``

        Returns:
            HDF5ResultsStore

        """
        result = result.get('record', result)
        with self as store:
            tasks = pipeline.tasks.values()
            tables = create_task_tables(self.h5file, tasks, result,
                                        overwrite=True)
            store._write_row(store.h5file, result, tables)

            # TODO: each task should have some description from func
            #       to write for metadata

        return self

# METADATA
    @property
    def basename(self):
        return os.path.basename(self.filename)

    @property
    def _attrs(self):
        with self as store:
            return self.h5file.root._v_attrs

    @property
    def attrs(self):
        return dict([(key, self._attrs[key]) for key
                     in self._attrs._v_attrnames])

    def set_attr(self, key, value):
        assert isinstance(value, six.string_types)
        self._attrs[key] = value

    def set_attrs(self, **attrs):
        for key, value in attrs.items():
            self.set_attr(key, value)

# GIS METADATA
    @property
    def crs(self):
        _crs = getattr(self._attrs, 'crs', '')
        if not _crs:
            raise AttributeError("Can't find 'crs' in result file")
        return CRS.from_string(_crs)

    @property
    def bounds(self):
        return BoundingBox(*self.bounding_box.bounds)

    @property
    def bounding_box(self):
        _bounds = getattr(self._attrs, 'bounds', '')
        if not _bounds:
            raise AttributeError("Can't find 'bounds' in result file")
        return shapely.wkt.loads(self._attrs['bounds'])

    @staticmethod
    def reader_attrs(reader, window):
        """ Extract reader attributes relevant to HDF5ResultsStore
        """
        crs = reader.crs.to_string()
        bounds = bounds_to_polygon(reader.window_bounds(window)).wkt
        return dict(crs=crs, bounds=bounds)

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
                dirname = os.path.dirname(self.filename)
                if dirname:
                    os.makedirs(dirname)
            except OSError as er:
                if er.errno == errno.EEXIST:
                    pass
                else:
                    raise

        self.h5file = tb.open_file(self.filename, mode=self.mode, title='YATSM',
                                   **self.tb_kwds)

        if (self._crs is not None and self._bounds is not None):
            self.set_attr('crs', self._crs)
            self.set_attr('bounds', self._bounds)

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

    def __getitem__(self, key):
        with self as store:
            key = key if key.startswith('/') else '/' + key
            if key not in store.keys():
                raise KeyError('Cannot find node {} in HDF5 store'.format(key))
            return store.h5file.get_node(key)

    def __setitem__(self, key, value):
        with self as store:
            if key not in store.keys():
                raise KeyError('Cannot find node {} in HDF5 store'.format(key))
            group, table = key.rsplit('/', 1)

            table = store[key]
            if not isinstance(table, tb.Table):
                raise AttributeError('Cannot set value for non-table '
                                     '{}'.format(key))
            else:
                table.append(value)

    def __repr__(self):
        return ("<{0.__class__.__name__}(filename={0.filename}, mode={0.mode})>"
                .format(self))
