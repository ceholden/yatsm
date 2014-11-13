#!/usr/bin/env python
""" Make changemap of YATSM output for a given date

Usage:
    yatsm_changemap.py [options] (first | last | num) <start_date> <end_date>
        <output>

Options:
    --root <dir>            Root time series directory [default: ./]
    -r --result <dir>       Directory of results [default: YATSM]
    -i --image <image>      Example image [default: example_img]
    --ndv <NoDataValue>     No data value for map [default: 0]
    -f --format <format>    Output raster format [default: GTiff]
    --date <format>         Input date format [default: %Y-%m-%d]
    --warn-on-empty         Warn user when reading in empty result files
    --version               Show program version
    -v --verbose            Show verbose debugging messages
    -h --help               Show help messages

Change date options:
    --magnitude             Add magnitude of change as extra bands
    --out_date <format>     Output date format [default: %Y%j]

Examples:
    # TODO

"""
from __future__ import division, print_function

from datetime import datetime as dt
import fnmatch
import logging
import os
import sys

from docopt import docopt
import numpy as np
from osgeo import gdal, gdal_array

# Handle runnin as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.utils import find_results, iter_records

gdal.UseExceptions()
gdal.AllRegister()

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

# Filters for results
_result_record = 'yatsm_*'

WARN_ON_EMPTY = False


# PROCESSING
def get_datechangemap(start, end, result_location, image_ds,
                      first=False,
                      out_format='%Y%j',
                      magnitude=False,
                      ndv=-9999, pattern=_result_record):
    """ Output raster with changemap

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      first (bool): Use first change instead of last
      out_format (str, optional): Output date format
      magnitude (bool, optional): output magnitude of each change?
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      raster (np.array): 3D numpy array containing the changes between the
        start and end date, and the change magnitude of each change if
        specified

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records):

        index = np.where((rec['break'] > 0) &
                         (rec['start'] >= start) &
                         (rec['end'] <= end))[0]
        if first:
            _, _index = np.unique(rec['px'][index], return_index=True)
            index = index[_index]

        if index.shape[0] != 0:
            if out_format != 'ordinal':
                dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                  for _d in rec['break'][index]])
                raster[rec['py'][index], rec['px'][index]] = dates
            else:
                raster[rec['py'][index], rec['px'][index]] = \
                    rec['break'][index]

    return raster


def get_numchangemap(start, end, result_location, image_ds,
                     ndv=-9999, pattern=_result_record):
    """ Output raster with changemap

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      result_location (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      raster (np.array): 2D numpy array containing the number of changes
        between the start and end date

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.int32) * int(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records):
        # X location of each changed model
        px_changed = rec['px'][(rec['break'] >= start) & (rec['break'] <= end)]
        # Count occurrences of changed pixel locations
        bincount = np.bincount(px_changed)
        # How many changes for unique values of px_changed?
        n_change = bincount[np.nonzero(bincount)[0]]

        # Add in the values
        px = np.unique(px_changed)
        py = rec['py'][np.in1d(px, rec['px'])]
        raster[py, px] = n_change

    return raster

def write_output(raster, output, image_ds, gdal_frmt, ndv, band_names=None):
    """ Write raster to output file """
    logger.debug('Writing output to disk')

    driver = gdal.GetDriverByName(gdal_frmt)

    if len(raster.shape) > 2:
        nband = raster.shape[2]
    else:
        nband = 1

    ds = driver.Create(
        output,
        image_ds.RasterXSize, image_ds.RasterYSize, nband,
        gdal_array.NumericTypeCodeToGDALTypeCode(raster.dtype.type)
    )

    if band_names is not None:
        if len(band_names) != nband:
            logger.error('Did not get enough names for all bands')
            sys.exit(1)

    if raster.ndim > 2:
        for b in range(nband):
            logger.debug('    writing band {b}'.format(b=b + 1))
            ds.GetRasterBand(b + 1).WriteArray(raster[:, :, b])
            ds.GetRasterBand(b + 1).SetNoDataValue(ndv)

            if band_names is not None:
                ds.GetRasterBand(b + 1).SetDescription(band_names[b])
    else:
        logger.debug('    writing band')
        ds.GetRasterBand(1).WriteArray(raster)
        ds.GetRasterBand(1).SetNoDataValue(ndv)

        if band_names is not None:
            ds.GetRasterBand(1).SetDescription(band_names[0])

    ds.SetProjection(image_ds.GetProjection())
    ds.SetGeoTransform(image_ds.GetGeoTransform())

    ds = None


def parse_args(args):
    """ Returns dictionary of parsed and validated command arguments

    Args:
      args (dict): Arguments from user

    Returns:
      dict: Parsed and validated arguments

    """
    parsed_args = { }
    ### Parse required input
    parsed_args['first'] = args['first']
    parsed_args['last'] = args['last']
    parsed_args['num'] = args['num']

    parsed_args['date_format'] = args['--date']
    # Start date for map
    start = args['<start_date>']
    try:
        start = dt.strptime(start, parsed_args['date_format'])
    except:
        logger.error('Could not parse start date {d} with format f'.format(
            d=start, f=parsed_args['date_format']))
        raise
    start = start.toordinal()

    # End date for map
    end = args['<end_date>']
    try:
        end = dt.strptime(end, parsed_args['date_format'])
    except:
        logger.error('Could not parse end date {d} with format f'.format(
            d=start, f=parsed_args['date_format']))
        raise
    end = end.toordinal()

    if start >= end:
        logger.error('Start date cannot be later than or equal to end')
        sys.exit(1)
    logger.debug('Making map of changes over {n} days'.format(n=end-start))

    parsed_args['start'] = start
    parsed_args['end'] = end

    # Output name
    parsed_args['output'] = os.path.abspath(args['<output>'])
    if not os.path.isdir(os.path.dirname(parsed_args['output'])):
        try:
            os.makedirs(os.path.dirname(parsed_args['output']))
        except:
            logger.error('Could not make output directory specified')
            raise

    ### Map options
    magnitude = args['--magnitude']
    if magnitude:
        raise NotImplementedError("Haven't written magnitude parser yet")

    ### Input options
    # Root directory
    root = args['--root']
    if not os.path.isdir(root):
        logger.error('Root directory is not a directory')
        sys.exit(1)

    # Results folder
    results = args['--result']
    # First look as relative path under root folder, then from PWD
    if not os.path.isdir(os.path.join(root, results)):
        if os.path.isdir(results):
            results = os.path.abspath(results)
        else:
            logger.error('Cannot find results folder')
            sys.exit(1)
    else:
        results = os.path.abspath(os.path.join(root, results))

    parsed_args['root'] = root
    parsed_args['results'] = results

    # Example image
    image = args['--image']
    if not os.path.isfile(image):
        if os.path.isfile(os.path.join(root, image)):
            image = os.path.join(root, image)
        else:
            logger.error('Cannot find example image')
            sys.exit(1)
    parsed_args['image'] = os.path.abspath(image)

    ### Output options
    out_format = args['--out_date']
    if out_format != 'ordinal':
        try:
            test = dt.today().strftime(out_format)
        except:
            logger.error('Could not parse output date format')
            raise
        try:
            test = int(test)
        except ValueError:
            logger.error('Cannot use output date format which uses characters')
            sys.exit(1)
        logger.debug('Outputting map using format {f}'.format(f=out_format))
    else:
        logger.debug('Outputting map using ordinal dates')
    parsed_args['out_format'] = out_format

    # NDV
    try:
        parsed_args['ndv'] = float(args['--ndv'])
    except ValueError:
        logger.error('NoDataValue must be a real number')
        raise

    # Raster file format
    parsed_args['gdal_frmt'] = args['--format']
    try:
        _ = gdal.GetDriverByName(parsed_args['gdal_frmt'])
    except:
        logger.error('Unknown GDAL format specified')
        raise

    return parsed_args


def main(args):
    """ Make change maps from parsed user inputs """
    ### Make map
    try:
        image_ds = gdal.Open(args['image'], gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    start_txt = dt.fromordinal(args['start']).strftime('%Y%j')
    end_txt = dt.fromordinal(args['end']).strftime('%Y%j')

    # Make map of date of change
    if args['first'] or args['last']:
        changemap = get_datechangemap(
            args['start'], args['end'], args['results'], image_ds,
            first=args['first'], out_format=args['out_format'],
            magnitude=None,
            ndv=args['ndv'], pattern=_result_record
        )

        band_names = ['ChangeDate_s{s}-e{e}'.format(s=start_txt, e=end_txt)]

    elif args['num']:
        changemap = get_numchangemap(
            args['start'], args['end'], args['results'], image_ds,
            ndv=args['ndv'], pattern=_result_record
        )
        band_names = ['NumChanges_{s}-e{e}'.format(s=start_txt, e=end_txt)]

    write_output(changemap, args['output'], image_ds,
                 args['gdal_frmt'], args['ndv'],
                 band_names=band_names)

    image_ds = None


if __name__ == '__main__':
    args = docopt(__doc__, version=__version__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)
    if args['--warn-on-empty']:
        WARN_ON_EMPTY = True

    args = parse_args(args)
    main(args)
