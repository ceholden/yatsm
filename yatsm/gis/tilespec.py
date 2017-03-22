""" Predefined tile specifications and utilities for working with tile systems
"""
import inspect
import itertools
import logging
import json
from pathlib import Path

from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
import shapely.geometry
import six

from .utils import bounds_to_polygon

logger = logging.getLogger(__name__)


class TileSpec(object):
    """ A tile specification or tile scheme

    Args:
        ul (tuple): upper left X/Y coordinates
        crs (dict): ``rasterio`` compatible coordinate system reference dict
        res (tuple): pixel X/Y resolution
        size (tuple): number of pixels in X/Y dimension of each tile
        desc (str): description of tile specification (default: None)
    """

    def __init__(self, ul, crs, res, size, desc=None):
        self.ul = ul
        if isinstance(crs, six.string_types):
            self.crs = CRS.from_string(crs)
        elif isinstance(crs, int):
            self.crs = CRS.from_epsg(crs)
        else:
            self.crs = crs

        if not self.crs.is_valid:
            raise ValueError('Could not parse coordinate reference system '
                             'string to a valid projection ({})'.format(crs))

        self.crs_str = self.crs.to_string()
        self.res = res
        self.size = size
        self.desc = desc or 'unnamed'
        self._tiles = {}

    def __repr__(self):
        return (
            "<{0.__class__.__name__}(desc={0.desc}, ul={0.ul}, crs={0.crs}, "
            "res={0.res}, size={0.size}) at {hex}>"
            .format(self, hex=hex(id(self)))
        )

    def __getitem__(self, index):
        """ Return a Tile for the grid row/column specified by index
        """
        if isinstance(index, tuple):
            if len(index) > 2:
                raise IndexError('TileSpec only has two dimensions (row/col)')
            if not isinstance(index[0], int) and isinstance(index[1], int):
                raise TypeError(
                    'Only support indexing int/int for now')
            return self._index_to_tile(index)
        else:
            raise IndexError('Unknown index type')

    def _index_to_bounds(self, index):
        """ Return Tile footprint bounds for given index

        Args:
            index (tuple): tile row/column index

        Returns:
            BoundingBox: the :attr:`BoundingBox` of a tile
        """
        return BoundingBox(
            left=self.ul[0] + index[1] * self.size[0] * self.res[0],
            right=self.ul[0] + (index[1] + 1) * self.size[0] * self.res[0],
            top=self.ul[1] - index[0] * self.size[1] * self.res[1],
            bottom=self.ul[1] - (index[0] + 1) * self.size[1] * self.res[1]
        )

    def _index_to_tile(self, index):
        """ Return the Tile for given index

        Args:
            index (tuple): tile row/column index

        Returns:
            Tile: a Tile object
        """
        if index not in self._tiles:
            bounds = self._index_to_bounds(index)
            self._tiles[index] = Tile(bounds, self.crs, index, self)
        return self._tiles[index]

    def bounds_to_tiles(self, bounds):
        """ Yield Tile objects for this TileSpec that intersect a given bounds

        .. note::

            It is required that the input ``bounds`` be in the same
            coordinate reference system as ``crs``.

        Args:
            bounds (BoundingBox): input bounds

        Yields:
            Tile: the Tiles that intersect within a bounds
        """
        grid_ys, grid_xs = self._frame_bounds(bounds)
        return self._yield_tiles(grid_ys, grid_xs, bounds)

    def point_to_tile(self, point):
        """ Return a :class:`Tile` containing a given point (x, y)

        Args:
            point (tuple): X/Y coordinates in tile specification's CRS
        Returns:
            Tile: The intersecting :class`Tile`
        """
        px, py = self.size[0] * self.res[0], self.size[1] * self.res[1]
        _x = int((point[0] - self.ul[0]) // px)
        _y = int((self.ul[1] - point[1]) // py)

        return self._index_to_tile((_y, _x))

    def roi_to_tiles(self, roi):
        """ Yield tiles overlapping a Region of Interest `shapely` geometry

        Args:
            roi (shapely.geometry.Polygon): A geometry in the tile
                specifications' crs
        Yields:
            Tile: A :class`Tile` that intersects the ROI
        """
        bounds = BoundingBox(roi.bounds)
        grid_ys, grid_xs = self._frame_bounds(bounds)
        return self._yield_tiles(grid_ys, grid_xs, bounds)

    def _yield_tiles(self, grid_ys, grid_xs, bounds):
        for index in itertools.product(grid_ys, grid_xs):
            tile = self._index_to_tile(index)
            if tile.polygon.intersects(bounds_to_polygon(bounds)):
                yield tile

    def _frame_bounds(self, bounds):
        px, py = self.size[0] * self.res[0], self.size[1] * self.res[1]
        min_grid_x = int((bounds.left - self.ul[0]) // px)
        max_grid_x = int((bounds.right - self.ul[0]) // px)
        min_grid_y = int((self.ul[1] - bounds.top) // py)
        max_grid_y = int((self.ul[1] - bounds.bottom) // py)
        return (range(min_grid_y, max_grid_y + 1),
                range(min_grid_x, max_grid_x + 1))


class Tile(object):
    """ A tile

    Args:
        bounds (BoundingBox): the bounding box of the tile
        crs (str): the coordinate reference system of the tile
        index (tuple): the index of this tile in the larger tile specification
        tilespec (TileSpec): the tile specification

    """

    def __init__(self, bounds, crs, index, tilespec):
        self.bounds = bounds
        self.crs = crs
        self.index = index
        self.tilespec = tilespec

    @property
    def vertical(self):
        """ int: The horizontal index of this tile in its tile specification
        """
        return self.index[0]

    @property
    def horizontal(self):
        """ int: The horizontal index of this tile in its tile specification
        """
        return self.index[1]

    @property
    def transform(self):
        """ Affine: The ``Affine`` transform for the tile
        """
        return Affine(self.tilespec.res[0], 0, self.bounds.left,
                      0, -self.tilespec.res[1], self.bounds.top)

    @property
    def polygon(self):
        """ shapely.geometry.Polygon: This tile's geometry
        """
        return bounds_to_polygon(self.bounds)

    @property
    def geojson(self):
        """ str: This tile's geometry and crs represented as GeoJSON
        """
        return {
            'type': 'Feature',
            'properties': {
                'horizontal': self.horizontal,
                'vertical': self.vertical
            },
            'geometry': shapely.geometry.mapping(self.polygon)
        }

    def str_format(self, s):
        """ Return a string .format'd with tile attributes

        Args:
            s (s): A string with format-compatible substitution fields

        Returns:
            str: A formatted string
        """
        attrs = {
            k: v for k, v in inspect.getmembers(self)
            if not callable(v) and not k.startswith('_')
        }
        return s.format(**attrs)


# Load tile specifications from package data
def retrieve_tilespecs(filename=None):
    """ Retrieve default tile specifications packaged within ``tilezilla``

    Args:
        filename (str): Optionally, override default data location

    Returns:
        dict: default tilespecs packaged within ``tilezilla`` as TileSpec
            objects
    """
    filename = filename or Path(__file__).parent.joinpaths('tile_specs.json')

    with open(filename, 'r') as f:
        tilespecs = json.load(f)

    for key in tilespecs:
        tilespecs[key]['crs'] = CRS.from_string(tilespecs[key]['crs'])
        tilespecs[key] = TileSpec(desc=key, **tilespecs[key])
    return tilespecs


#: dict: Built-in tile specifications available by default
TILESPECS = retrieve_tilespecs()
