""" Command line interface for creating changemaps of YATSM algorithm output
"""
from datetime import datetime as dt
import logging
import os


import click
import numpy as np
from osgeo import gdal

from yatsm.cli import options
from yatsm.utils import find_results, iter_records, write_output

gdal.AllRegister()
gdal.UseExceptions()

logger = logging.getLogger('yatsm')

# Filters for results
_result_record = 'yatsm_*'

WARN_ON_EMPTY = False


@click.command(
    short_help='Map change found by YATSM algorithm over time period')
@click.argument('map_type', metavar='<map_type>',
                type=click.Choice(['first', 'last', 'num']))
@options.arg_date(var='start_date', metavar='<start_date>')
@options.arg_date(var='end_date', metavar='<end_date>')
@options.arg_output
@options.opt_rootdir
@options.opt_resultdir
@options.opt_exampleimg
@options.opt_date_format
@options.opt_nodata
@options.opt_format
@click.option('--out_date', 'out_date_frmt', metavar='<format>',
              default='%Y%j', show_default=True, help='Output date format')
@click.option('--warn-on-empty', is_flag=True,
              help='Warn user when reading in empty results files')
@click.option('--magnitude', is_flag=True,
              help='Add magnitude of change as extra image '
                   '(pattern is [name]_mag[ext])')
@click.pass_context
def changemap(ctx, map_type, start_date, end_date, output,
              root, result, image, date_frmt, ndv, gdal_frmt, out_date_frmt,
              warn_on_empty, magnitude):
    """
    Examples: TODO
    """
    gdal_frmt = str(gdal_frmt)  # GDAL GetDriverByName doesn't work on Unicode

    frmt = '%Y%j'
    start_txt, end_txt = start_date.strftime(frmt), end_date.strftime(frmt)
    start_date, end_date = start_date.toordinal(), end_date.toordinal()

    try:
        image_ds = gdal.Open(image, gdal.GA_ReadOnly)
    except:
        logger.error('Could not open example image for reading')
        raise

    if map_type in ('first', 'last'):
        changemap, magnitudemap, magnitude_indices = get_datechangemap(
            start_date, end_date, result, image_ds,
            first=map_type == 'first', out_format=out_date_frmt,
            magnitude=magnitude,
            ndv=ndv, pattern=_result_record
        )

        band_names = ['ChangeDate_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

        if magnitudemap is not None:
            band_names = (['Magnitude Index {}'.format(i) for
                           i in magnitude_indices])
            name, ext = os.path.splitext(output)
            output = name + '_mag' + ext
            write_output(magnitudemap, output, image_ds, gdal_frmt, ndv,
                         band_names=band_names)

    elif map_type == 'num':
        changemap = get_numchangemap(
            start_date, end_date, result, image_ds,
            ndv=ndv, pattern=_result_record
        )

        band_names = ['NumChanges_s{s}-e{e}'.format(s=start_txt, e=end_txt)]
        write_output(changemap, output, image_ds, gdal_frmt, ndv,
                     band_names=band_names)

    image_ds = None


# UTILITIES
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
            raise click.Abort()

        changed = np.where(rec_array['break'] != 0)[0]
        if changed.size == 0:
            continue

        logger.debug('Using non-zero elements of "magnitude" field in '
                     'changed records for magnitude indices')
        return np.nonzero(np.any(rec_array[changed]['magnitude'] != 0))[0]


# MAPPING FUNCTIONS
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
