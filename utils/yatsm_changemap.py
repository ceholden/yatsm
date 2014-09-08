#!/usr/bin/env python
""" Make changemap of YATSM output for a given date

Usage:
    yatsm_changemap.py [options] <start_date> <end_date> <output>

Options:
    --out_date <format>     Output date format [default: %Y%j]
    --firstchange           Show first change instead of last
    --samechange <lut>      Use LUT to eliminate changes to/from same class
    --ndv <NoDataValue>     No data value for map [default: 0]
    -d --directory <dir>    Root time series directory [default: ./]
    -r --result <dir>       Directory of results [default: YATSM]
    -i --image <image>      Example image [default: example_img]
    --date <format>         Date format [default: %Y-%m-%d]
    -f --format <format>    Output raster format [default: GTiff]
    -v --verbose            Show verbose debugging messages
    -h --help               Show help messages

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

gdal.UseExceptions()
gdal.AllRegister()

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
                    level=logging.INFO,
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Filters for results
_result_record = 'yatsm_*'


# UTILITY FUNCTIONS
def find_results(location, pattern):
    """ Create list of result files and return sorted

    Args:
      location (str): directory location to search
      pattern (str): glob style search pattern for results

    Returns:
      results (list): list of file paths for results found

    """
    # Note: already checked for location existence in main()
    records = []
    for root, dirnames, filenames in os.walk(location):
        for filename in fnmatch.filter(filenames, pattern):
            records.append(os.path.join(root, filename))

    if len(records) == 0:
        logger.error('Error: could not find results in: {0}'.format(location))
        sys.exit(1)

    records.sort()

    return records


# PROCESSING
def get_changemap(start, end, results, image_ds,
                  out_format='%Y%j',
                  first=False, lut=None,
                  ndv=-9999, pattern=_result_record):
    """ Output raster with changemap

    Args:
      start (int): Ordinal date for start of map records
      end (int): Ordinal date for end of map records
      results (str): Location of results
      image_ds (gdal.Dataset): Example dataset
      out_format (str, optional): Output date format
      first (bool): Use first change instead of last
      lut (np.array): 2 column lookup table for determining "same change"
      ndv (int, optional): NoDataValue
      pattern (str, optional): filename pattern of saved record results

    Returns:
      raster (np.array): 2D numpy array containing the changes between the
        start and end date

    """
    # Find results
    records = find_results(results, pattern)
    n_records = len(records)

    raster = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                     dtype=np.int32) * ndv

    logger.debug('Processing results')
    for _i, r in enumerate(records):
        if np.mod(_i, 100) == 0:
            logger.debug('{0:.0f}%'.format(_i / n_records * 100))

        # Open output
        try:
            rec = np.load(r)['record']
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(f=r))
            continue

        if rec.shape[0] == 0:
            # No values in this file
            logger.warning('Could not find results in {f}'.format(f=r))
            continue

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


def main():
    """ Test input and pass to appropriate functions """
    ### Parse required input
    date_format = args['--date']
    # Start date for map
    start = args['<start_date>']
    try:
        start = dt.strptime(start, date_format)
    except:
        logger.error('Could not parse start date')
        raise
    start = start.toordinal()

    # End date for map
    end = args['<end_date>']
    try:
        end = dt.strptime(end, date_format)
    except:
        logger.error('Could not parse end date')
        raise
    end = end.toordinal()

    if start >= end:
        logger.error('Start date cannot be later than or equal to end')
        sys.exit(1)

    logger.debug('Making map of changes over {n} days'.format(n=end-start))

    # Output name
    output = os.path.abspath(args['<output>'])
    if not os.path.isdir(os.path.dirname(output)):
        try:
            os.makedirs(os.path.dirname(output))
        except:
            logger.error('Could not make output directory specified')
            raise

    ### Parse map options
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

    first = args['--firstchange']
    lut = args['--samechange']
    if lut:
        raise NotImplementedError("Haven't written LUT parser")

    ### Parse generic options
    # NDV
    try:
        ndv = float(args['--ndv'])
    except ValueError:
        logger.error('NoDataValue must be a real number')
        raise

    # Root directory
    root = args['--directory']
    if not os.path.isdir(root):
        logger.error('Root directory is not a directory')
        sys.exit(1)

    # Results folder
    results = args['--result']
    if not os.path.isdir(results):
        if os.path.isdir(os.path.join(root, results)):
            results = os.path.join(root, results)
        else:
            logger.error('Cannot find results folder')
            sys.exit(1)
    results = os.path.abspath(results)

    # Example image
    image = args['--image']
    if not os.path.isfile(image):
        if os.path.isfile(os.path.join(root, image)):
            image = os.path.join(root, image)
        else:
            logger.error('Cannot find example image')
            sys.exit(1)
    image = os.path.abspath(image)

    # Raster file format
    gdal_frmt = args['--format']
    try:
        _ = gdal.GetDriverByName(gdal_frmt)
    except:
        logger.error('Unknown GDAL format specified')
        raise

    ### Produce output specified
    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    ### Make map
    changemap = get_changemap(start, end, results, image_ds,
                              out_format=out_format,
                              first=first, lut=lut,
                              ndv=ndv, pattern=_result_record)

    start_txt = dt.fromordinal(start).strftime('%Y%j')
    end_txt = dt.fromordinal(end).strftime('%Y%j')
    write_output(changemap, output, image_ds, gdal_frmt, ndv,
                 ['Change_s{s}-e{e}'.format(s=start_txt, e=end_txt)])

    image_ds = None


if __name__ == '__main__':
    args = docopt(__doc__)

    if args['--verbose']:
        logger.setLevel(logging.DEBUG)

    main()
