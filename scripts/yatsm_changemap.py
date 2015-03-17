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
    --magnitude             Add magnitude of change as extra image (pattern
                                is [name]_mag[ext])
    --out_date <format>     Output date format [default: %Y%j]

Examples:
    # TODO

"""
from __future__ import division, print_function

from datetime import datetime as dt
import logging
import os
import sys

from docopt import docopt
import numpy as np
from osgeo import gdal

# Handle running as installed module or not
try:
    from yatsm.version import __version__
except ImportError:
    # Try adding `pwd` to PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from yatsm.version import __version__
from yatsm.utils import find_results, iter_records, write_output

FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')

gdal.AllRegister()
gdal.UseExceptions()

# Filters for results
_result_record = 'yatsm_*'

WARN_ON_EMPTY = False


# PROCESSING
def get_magnitude_indices(results):
    """ Finds indices of result containing magnitude of change information

    Args:
      results (iterable): list of result files to check within

    Returns:
      np.ndarray: indices containing magnitude change information from the
        tested indices

    """
    for result in results:
        try:
            rec = np.load(result)
        except (ValueError, AssertionError):
            logger.warning('Error reading {f}. May be corrupted'.format(
                f=result))
            continue

        # First search for record of `test_indices`
        if 'test_indices' in rec.files:
            logger.debug('Using `test_indices` information for magnitude')
            return rec['test_indices']

        # Fall back to using non-zero elements of 'record' record array
        rec_array = rec['record']
        if rec_array.dtype.names is None:
            # Empty record -- skip
            continue

        if 'magnitude' not in rec_array.dtype.names:
            logger.error('Cannot map magnitude of change')
            logger.error('Magnitude information not present in file {f} -- '
                'has it been calculated?'.format(f=result))
            logger.error('Version of result file: {v}'.format(
                v=rec['version'] if 'version' in rec.files else 'Unknown'))
            sys.exit(1)

        changed = np.where(rec_array['break'] != 0)[0]
        if changed.size == 0:
            continue

        logger.debug('Using non-zero elements of "magnitude" field in '
                     'changed records for magnitude indices')
        return np.nonzero(np.any(rec_array[changed]['magnitude'] != 0))[0]


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
      tuple: A 2D np.ndarray array containing the changes between the
        start and end date. Also includes, if specified, a 3D np.ndarray of
        the magnitude of each change plus the indices of these magnitudes

    """
    # Find results
    records = find_results(result_location, pattern)

    logger.debug('Allocating memory...')
    datemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize),
                      dtype=np.int32) * int(ndv)
    # Determine what magnitude information to output if requested
    if magnitude:
        magnitude_indices = get_magnitude_indices(records)
        magnitudemap = np.ones((image_ds.RasterYSize, image_ds.RasterXSize,
                               magnitude_indices.size),
                               dtype=np.float32) * float(ndv)

    logger.debug('Processing results')
    for rec in iter_records(records):

        index = np.where((rec['break'] >= start) &
                         (rec['break'] <= end))[0]

        if first:
            _, _index = np.unique(rec['px'][index], return_index=True)
            index = index[_index]

        if index.shape[0] != 0:
            if out_format != 'ordinal':
                dates = np.array([int(dt.fromordinal(_d).strftime(out_format))
                                  for _d in rec['break'][index]])
                datemap[rec['py'][index], rec['px'][index]] = dates
            else:
                datemap[rec['py'][index], rec['px'][index]] = \
                    rec['break'][index]
            if magnitude:
                magnitudemap[rec['py'][index], rec['px'][index], :] = \
                    rec[index]['magnitude'][:, magnitude_indices]

    if magnitude:
        return datemap, magnitudemap, magnitude_indices
    else:
        return datemap, None, None


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
      np.ndarray: 2D numpy array containing the number of changes
        between the start and end date; list containing band names

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


def parse_args(args):
    """ Returns dictionary of parsed and validated command arguments

    Args:
      args (dict): Arguments from user

    Returns:
      dict: Parsed and validated arguments

    """
    parsed_args = {}
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
    parsed_args['magnitude'] = args['--magnitude']

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
        gdal.GetDriverByName(parsed_args['gdal_frmt'])
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
        changemap, magnitudemap, magnitude_indices = get_datechangemap(
            args['start'], args['end'], args['results'], image_ds,
            first=args['first'], out_format=args['out_format'],
            magnitude=args['magnitude'],
            ndv=args['ndv'], pattern=_result_record
        )

        band_names = ['ChangeDate_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, args['output'], image_ds,
                     args['gdal_frmt'], args['ndv'],
                     band_names=band_names)

        if magnitudemap is not None:
            band_names = (['Magnitude Index ' + str(i) for i in
                          magnitude_indices])
            name, ext = os.path.splitext(args['output'])
            output = name + '_mag' + ext
            write_output(magnitudemap, output, image_ds,
                         args['gdal_frmt'], args['ndv'],
                         band_names=band_names)

    elif args['num']:
        changemap = get_numchangemap(
            args['start'], args['end'], args['results'], image_ds,
            ndv=args['ndv'], pattern=_result_record
        )

        band_names = ['NumChanges_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
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
