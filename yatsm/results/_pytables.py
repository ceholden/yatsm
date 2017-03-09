""" Results storage in HDF5 datasets using PyTables
"""
import datetime as dt
import fnmatch
import logging
import os
from pathlib import Path
import re

import numpy as np
import six
import tables as tb

from yatsm.algorithms import SEGMENT_ATTRS
from yatsm.gis import Georeference
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
        _dt, _ = dtype.fields[name]
        if issubclass(_dt.type, np.datetime64):
            tb_dtype = tb.Description({name: tb.Time64Col(pos=idx)})
        else:
            tb_dtype, byteorder = tb.descr_from_dtype(np.dtype([(name, _dt)]))
        _tb_dtype = tb_dtype._v_colobjects
        _tb_dtype[name]._v_pos = idx
        desc.update(_tb_dtype)
    return desc


def read_where(table, condition, fields, out=None, **where_kwds):
    """ A better version of `table.read_where` that accepts multiple fields

    Args:
        table (tb.Table): Table
        condition (str): Search condition
        fields (list[str]): Fields to return
        where_kwds (dict): Keyword options to pass to :ref:`tables.Table.where`
            and other similar functions

    Returns:
        np.ndarray: Structured array of results
    """
    idx = table.get_where_list(condition, **where_kwds)

    _dtype = np.dtype([(col, table.coldtypes[col]) for col in fields])
    if out is None:
        out = np.empty(idx.size, dtype=_dtype)
    else:
        for _descr in _dtype.descr:
            if _descr not in out.dtype.descr:
                raise TypeError('Provided workspace array "out" is '
                                'incompatible with required datatypes: {}'
                                .format(_dtype.descr))
        logger.debug('Using provided `out` workspace array. TODO check compat')

    for col in fields:
        out[col] = table.read_coordinates(idx, field=col)
    return out


def create_table(h5file, where, name, result, attrs=None, georef=None,
                 index=True, expectedrows=10000, overwrite=False,
                 **table_config):
    """ Create table to store results

    Args:
        h5file (tables.file.File): PyTables HDF5 file
        where (str or tables.group.Group): Parent group to place table
        name (str): Name of new table
        result (np.ndarray): Results as a NumPy structured array
        attrs (dict): Metadata to store as ``table.attrs``
        georef (Georeference): Georeferencing information to add to
            ``table.attrs``
        index (bool): Create index on :ref:`SEGMENT_ATTRS`
        expectedrows (int): Expected number of rows to store in table
        overwrite (bool): Overwrite existing table
        table_config (dict): Additional keyword arguments to be passed
            to ``h5file.create_table``

    Returns:
        table.table.Table: HDF5 table
    """
    table_desc = dtype_to_table(result.dtype)

    if _has_node(h5file, where, name=name) and not overwrite:
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

        if attrs:
            for key, value in attrs.items():
                table.attrs[key] = attrs[key]
        if georef:
            table = georeference(table, georef)

    return table


def georeference(node, georef):
    """ Georeference a :class:`tables.File`

    Args:
        h5file (tables.File): Open HDF5 file
        georef (Georeference): Georeference information
        where (str): HDF5 file node location

    Returns:
        tables.Node: Georeferenced HDF5 node
    """
    # TODO: decide between these approaches

    # 1. Store strings individually
    for key, val in georef.str.items():
        logger.debug('Setting %s to %s' % (key, val))
        node._v_attrs[key] = val

    # 2. Store all as JSON
    node._v_attrs['georef'] = georef.to_json()

    return node


def get_georeference(node):
    """ Get georeferencing information for a tag

    Args:
        node (tables.Node): HDF5 node

    Returns:
        Georeference: Georeferencing information

    Raises:
        KeyError: Raise if georeferencing is missing
    """
    # TODO: decide between these approaches

    # 1. Strings stored individually
    args = []
    for key in Georeference._fields:
        if key not in node._v_attrs._v_attrnames:
            raise KeyError('Missing georeferencing attr "%s" in %r' %
                           (key, node))
        value = node._v_attrs[key]
        if isinstance(value, np.string_):
            logger.debug('Decoding string')
            value = value.decode('utf-8')
        args.append(value)
    georef = Georeference.from_strings(*args)

    # 2. Store all as JSON
    try:
        if 'georef' not in node._v_attrs:
            if node._v_file.mode in ('w', 'r+') and node._v_file.isopen:
                logger.warning('Temporarily adding "georef" to your file...')
                node._v_attrs['georef'] = georef.to_json()
        if 'georef' in node._v_attrs:
            georef = Georeference.from_json(node._v_attrs['georef'])
    except Exception as e:
        from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA

    return georef


class HDF5ResultsStore(object):
    """ PyTables based HDF5 results storage

    Two ways to open and begin reading or writing to the file.

    Using a context manager:

    .. code-block:: python

        >>> with HDF5ResultsStore('some_data.h5', 'r+') as store:
        >>>     print(store.georef)
        >>>     print(store['/result/table', 'column'])

    Explicitly calling :ref:`~HDF5ResultsStore.start` and
    :ref:`HDF5ResultsStore.close()`:

    .. code-block:: python

        >>> store = HDF5ResultsStore('some_data.h5', 'r+').start()
        >>> print(store.georef)
        >>> print(store['/result/table', 'column'])
        >>> store.close()

    Args:
        filename (str): HDF5 file
        mode (str): File mode to open with. By default, opens in read mode or
            write mode if file doesn't exist
        georef (Georeference): HDF file's georeference information
        title (str): Title of HDF5 file
        keep_open (bool): Keep file handle open after calls
        overwrite (bool): Overwrite file attributes or data
        tb_kwds: Optional keywork arguments to :ref:`tables.open_file`
    """

    def __init__(self, filename, mode=None, georef=None,
                 title='YATSM',
                 keep_open=True, overwrite=False, **tb_kwds):
        _exists = os.path.exists(filename)

        self.filename = filename
        self.path = Path(filename)
        self.mode = mode or 'r' if _exists else 'w'
        self.georef = georef
        self.title = title
        self.keep_open = keep_open
        self.overwrite = overwrite
        self.tb_kwds = tb_kwds

        if self.mode == 'w' and _exists and not self.overwrite:
            logger.warning('Moving you down to "r+" permissions. '
                           'Use `overwrite=True` if you want to nuke '
                           'the preexisting file.')
            self.mode = 'r+'

        self.h5file = None
        if not _exists and not isinstance(self.georef, Georeference):
            raise TypeError('Must specify `georef` as `Georeference` when '
                            'creating a file')

# CREATION
    @classmethod
    def from_window(cls, window, reader=None, georef=None,
                    root='.', pattern=RESULT_TEMPLATE,
                    **open_kwds):
        """ Return instance of class for a given window

        When creating a file, the following attributes must be specified in
        one of two ways:

            1. Passing ``reader``
            2. Passing ``Georeference``

        They may either be explicitly passed, or retrieved from a reader
        instance from ``reader``

        Args:
            window (tuple): x_min, y_min, x_max, y_max for the given window
            reader (rasterio.DatasetReader)
            root (str): Root directory to save file
            pattern (str): Filename pattern to use, usually derived in part
                from attributes of ``window``

        Returns:
            cls: HDF5ResultsStore
        """
        filename = result_filename(window, root=root, pattern=pattern)

        georef = (georef if isinstance(georef, Georeference) else
                  Georeference.from_reader(reader))
        if not georef:
            raise TypeError('Must provide either `reader` or `georef`')

        return cls(filename,
                   georef=georef,
                   **open_kwds)

# READING
    def find_column(self, pattern, where='/', regex=False, return_table=False):
        """ Return :ref:`tb.Column` matching pattern
        """
        pattern = pattern if regex else fnmatch.translate(pattern)

        with self as store:
            for table in store.h5file.walk_nodes(where=where,
                                                 classname='Table'):
                colnames = [c for c in table.cols._v_colnames
                            if re.match(pattern, c)]
                for colname in colnames:
                    if return_table:
                        yield table, colname
                    else:
                        yield colname

    def query(self, name, columns=(),
              px=None, py=None, d_start=None, d_end=None, d_break=None,
              *query_terms):
        """ Yield table results from a search query

        For arguments where `slice` are possible arguments (coordinates and
        dates), passing a single value will construct a query using the
        following sign conventions:

            * px ==
            * py ==
            * d_start <
            * d_end >
            * d_break >

        Args:
            name (str): Name of table containg segment results
            column (str, or iterable): One or more columns to include in
                returned results
            px (float, or slice): One X coordinate, or a range of X coordinates
            px (float, or slice): One Y coordinate, or a range of Y coordinates
            d_start (datetime, or slice): One date, or a range of dates
            d_end (datetime, or slice): One date, or a range of dates
            d_break (datetime, or slice): One date, or a range of dates
            *query_terms: Additional search terms to send to ``Table.where``

        """
        # TODO: deal when we have a table join vs inside table
        # TODO: look into "condavars" argument
        if isinstance(columns, six.string_types):
            columns = (columns, )

        table = self[name]

        def _build_dt(name, sign, d):
            if isinstance(d, dt.datetime):
                # TODO: have attr on self to decide time storage convention
                return '(%s %s %d)' % (name, sign, d.toordinal())
            elif isinstance(d, slice):
                return ' & '.join([_build_dt(name, s, i) for (s, i) in
                                   (('start', d.start), ('stop', d.stop))])

        def _build_coord(name, sign, coord):
            if isinstance(coord, (float, int)):
                return '%s %s %d' % (name, sign, coord)
            elif isinstance(coord, slice):
                return ' & '.join([_build_coord(name, s, i) for (s, i) in
                                   (('start', coord.start),
                                    ('stop', coord.stop))])

        px = _build_coord('px', '==', px)
        py = _build_coord('py', '==', py)
        d_start = _build_dt('start_day', '>', d_start)
        d_end = _build_dt('end_day', '<', d_end)
        d_break = _build_dt('break_day', '>', d_break)

        query = ' & '.join([item for item in
                            (px, py, d_start, d_end, d_break, ) + query_terms
                            if item])

        logger.debug('Searching: %s' % query)
        if columns:
            return read_where(table, query, columns)
        else:
            return table.read_where(table, query)

# WRITING
    def write_result(self, pipeline, result, georef=None,
                     overwrite=None, **kwds):
        """ Write result to HDF5

        Args:
            pipeline (yatsm.pipeline.Pipeline): YATSM pipeline of tasks
            result (dict): Dictionary of pipeline 'record' results
                where key is task output and value is a structured
                :ref:`np.ndarray`
            georef (Georeference): Georeferencing information
            overwrite (bool): Overwrite existing values, overriding
                even the class preference (:ref:`HDF5ResultsStore.overwrite`).
                Defaults to behavior chosen during initialization

        Returns:
            HDF5ResultsStore
        """
        do_overwrite = self.overwrite if overwrite is None else overwrite
        with self as store:
            for task, (where, name) in pipeline.task_tables.items():
                if not where or not name:
                    continue
                table = create_table(store.h5file,
                                     where,
                                     name,
                                     result[task.output_record],
                                     attrs=task.metadata,
                                     georef=georef,
                                     overwrite=do_overwrite,
                                     **kwds)
                table.append(result[task.output_record])
                table.flush()

        return self

    def completed(self, pipeline, min_rows=1000):
        """ Return True if all tables from pipeline have been written

        Args:
            pipeline (yatsm.pipeline.Pipeline): Pipeline of tasks
            min_rows (int): Minimum number of rows to qualify as
                having been completed

        Returns:
            bool: True if it looks like the data have been written
        """
        with self as store:
            for (where, name) in pipeline.task_tables.values():
                if where and name:
                    try:
                        table = store.h5file.get_node(where, name)
                    except tb.NoSuchNodeError:
                        return False
                    else:
                        if not isinstance(table, tb.Table):
                            logger.debug('Node named "%s" is not a table' %
                                         name)
                            return False
                        if table.shape[0] < min_rows:
                            logger.debug('Table "%s" has fewer than %i rows' %
                                         (name, min_rows))
                            return False
        return True

# METADATA
    @property
    def basename(self):
        return os.path.basename(self.filename)

    @property
    def _tags(self):
        """ Alias of self.h5file.root._v_attrs
        """
        if self.h5file.isopen:
            return self.h5file.root._v_attrs

    @property
    def tags(self):
        with self as store:
            return dict([(key, store._tags[key]) for key
                         in store._tags._v_attrnames])

    def update_tag(self, key, value):
        assert isinstance(value, six.string_types)
        self._tags[key] = value

    def update_tags(self, **tags):
        for key, value in tags.items():
            self.update_tag(key, value)

# GIS METADATA
    def get_georef(self, where='/'):
        """ Return georeferencing info for a node or table
        """
        with self as store:
            node = store.h5file.get_node(where)
            return get_georeference(node)

    def set_georef(self, where='/', georef=None, **geo_kwds):
        """ Set file georeferencing information

        Args:
            where (str or tables.Node): Node location to georeference
            georef (Georeference): Georeferencing information
            geo_kwds (dict): Specific :class:`Georeferencing` attribute to
                modify (need to exist beforehand)

        Returns:
            tables.Node: Georeferenced HDF5 node
        """
        with self as store:
            node = store.h5file.get_node(where)
            # Get modified georef
            node_georef = get_georeference(node).modify(georef, **geo_kwds)
            # Modify it
            return georeference(node, node_georef)

    @property
    def crs(self):
        """ rasterio.crs.CRS: Coordinate reference system
        """
        with self as store:
            return store.georef.crs

    @property
    def bounds(self):
        """ BoundingBox: Bounding box of data in file
        """
        with self as store:
            return store.georef.bounds

    @property
    def transform(self):
        """ affine.Affine: Affine transform
        """
        with self as store:
            return store.georef.transform

    @property
    def bbox(self):
        """ shapely.geometry.Polygon: Bounding box as polygon
        """
        with self as store:
            return store.georef.bbox

# CONTEXT HELPERS
    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        if not self.keep_open:
            self.close()

    def __del__(self):
        self.close()

    def start(self):
        """ Begin reading/writing to file

        Returns:
            HDF5ResultsStore: Return the dataset storage, but opened
                according to :attr:`HDF5ResultsStore.mode`
        """
        if isinstance(self.h5file, tb.file.File):
            if (getattr(self.h5file, 'mode', '') == self.mode
                    and self.h5file.isopen):
                return self  # already opened in correct form, bail
            else:
                self.h5file.close()

        logger.debug('Opening %s in mode %s' % (self.filename, self.mode))
        self.h5file = tb.open_file(self.filename, mode=self.mode,
                                   title=self.title,
                                   **self.tb_kwds)

        # Set GIS related tags on write/overwrite
        if self.mode == 'w' or self.overwrite:
            georeference(self.h5file.root, self.georef)
        else:
            self.georef = get_georeference(self.h5file.root)
        return self

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
        """ Yields key/value pairs for nodes
        """
        with self as store:
            for node in store.h5file.walk_nodes():
                yield node._v_pathname, node

    def groups(self):
        """ Yields key/value pairs for :ref:`tables.Group` stored
        """
        for name, node in self.items():
            if isinstance(node, tb.Group):
                yield name, node

    def tables(self):
        """ Yields key/value pairs for :ref:`tables.Table` stored
        """
        with self as store:
            for node in store.h5file.walk_nodes(classname='Table'):
                yield node._v_pathname, node

    def get(self, key, default=None):
        """ Return value for `key` if exists otherwise `default`
        """
        with self as store:
            try:
                node = store.h5file.get_node(key)
            except tb.NoSuchNodeError:
                return default
            else:
                return node

    def __getitem__(self, key, col=None):
        """ Allow table/group access as str, or tuple that also gives Column
        """
        if not key:  # guard empty string, None, etc.
            raise KeyError('No such group or table "%s"' % key)

        with self as store:
            if isinstance(key, tuple):
                key, col = key
            key = key if key.startswith('/') else '/' + key

            if key not in store.keys():
                raise KeyError('Cannot find node {} in HDF5 store'.format(key))

            node = store.h5file.get_node(key)
            if col:
                if col not in node.cols._v_colnames:
                    raise KeyError('Cannot find column "{0}" in node "{1}"'
                                   .format(col, key))
                return store.h5file.get_node(key).cols._g_col(col)
            else:
                return node

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
        opened = self.h5file and self.h5file.isopen
        return ("{0} <{1.__class__.__name__}"
                "(filename={1.filename}, mode={1.mode})>"
                .format('Open' if opened else 'Closed', self))
