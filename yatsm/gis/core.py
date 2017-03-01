from collections import namedtuple
import re

import six

from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from shapely.geometry import Polygon

from yatsm.gis.convert import GIS_TO_STR, STR_TO_GIS
from yatsm.gis.utils import bounds_to_polygon


_georeference = namedtuple('Georeference',
                           ('crs', 'bounds', 'transform', 'bbox', ))


class _Georeference(_georeference):
    """ :class:`Georeference`'s parent who doesn't typecheck
    """
    def __repr__(self):
        summary = ['<yatsm.gis.%s at %s>' % (type(self).__name__,
                                             hex(id(self)))]
        indent = ' ' * 2

        summary.append('%s * crs: %s' % (indent, self.crs))
        summary.append('%s * bounds: %s' % (indent, self.bounds))
        summary.append('%s * transform: %s' % (
            indent,
            re.sub('\s+', ' ', repr(self.transform).strip('\n'))
        ))
        summary.append('%s * bbox: %s' % (indent, self.bbox))

        return '\n'.join(summary)

    def keys(self):
        for field in self._fields:
            yield field

    def items(self):
        for field, val in zip(self._fields, self):
            yield (field, val)


class Georeference(_Georeference):
    """ namedtuple holding georeferencing information
    """

    def __new__(cls, crs, bounds, transform, bbox):
        if isinstance(crs, six.string_types):
            crs = STR_TO_GIS(crs)
        if isinstance(bounds, six.string_types):
            bounds = STR_TO_GIS(bounds)
        if isinstance(transform, six.string_types):
            transform = STR_TO_GIS(transform)
        if isinstance(bbox, six.string_types):
            bbox = STR_TO_GIS(bbox)

        # TODO: real TypeError raising
        assert isinstance(crs, CRS)
        assert isinstance(bounds, BoundingBox)
        assert isinstance(transform, Affine)
        assert isinstance(bbox, Polygon)

        self = super(Georeference, cls).__new__(
            cls, crs, bounds, transform, bbox)
        return self

    @classmethod
    def from_reader(cls, reader):
        """ Construct a `Georeference` from a :class:`rasterio.DatasetReader`
        """
        return cls(crs=reader.crs,
                   bounds=reader.bounds,
                   transform=reader.transform,
                   bbox=bounds_to_polygon(reader.bounds))

    @classmethod
    def from_strings(cls, crs, bounds, transform, bbox):
        return cls(crs=STR_TO_GIS['crs'](crs),
                   bounds=STR_TO_GIS['bounds'](bounds),
                   transform=STR_TO_GIS['transform'](transform),
                   bbox=STR_TO_GIS['bbox'](bbox))

    @property
    def str(self):
        return _Georeference(
            *(GIS_TO_STR[field](getattr(self, field))
              for field in _Georeference._fields)
        )


if __name__ == '__main__':
    import rasterio
    s = rasterio.open('asdf.tif')
    g = Georeference.from_reader(s)
    from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA
