""" Command line interface for running YATSM on image lines """
import logging
import os
import time

import click
import numpy as np
import patsy

from yatsm.cache import test_cache
from yatsm.cli import options
from yatsm.config_parser import parse_config_file
import yatsm._cyprep as cyprep
from yatsm.errors import TSLengthException
from yatsm.utils import (distribute_jobs, get_output_name, get_image_IDs,
                         csvfile_to_dataset)
from yatsm.reader import get_image_attribute, read_line
from yatsm.regression.transforms import harm
from yatsm.yatsm import YATSM
try:
    import yatsm.phenology as pheno
except ImportError:
    pheno = None
from yatsm.version import __version__

logger = logging.getLogger('yatsm')
logger_algo = logging.getLogger('yatsm_algo')


@click.command(short_help='Run YATSM on an entire image line by line')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.option('--check', is_flag=True,
              help='Check that images exist')
@click.option('--check_cache', is_flag=True,
              help='Check that cache file contains matching data')
@click.option('--resume', is_flag=True,
              help='Do not overwrite preexisting results')
@click.option('--do-not-run', is_flag=True,
              help='Do not run YATSM (useful for just caching data)')
@click.option('--verbose-yatsm', is_flag=True,
              help='Show verbose debugging messages in YATSM algorithm')
@click.pass_context
def line(ctx, config, job_number, total_jobs,
         resume, check, check_cache, do_not_run, verbose_yatsm):
    if verbose_yatsm:
        logger_algo.setLevel(logging.DEBUG)

    # Parse config
    dataset_config, yatsm_config = parse_config_file(config)

    if yatsm_config.get('calc_pheno') and pheno is None:
        logger.error('Could not import yatsm.phenology but phenology metrics '
                     'are requested')
        raise click.Abort()

    # Make sure output directory exists and is writable
    try:
        os.makedirs(dataset_config['output'])
    except OSError as e:
        # File exists
        if e.errno == 17:
            pass
        elif e.errno == 13:
            logger.error('Cannot create output directory {d}'.format(
                d=dataset_config['output']))
            raise click.Abort()

    if not os.access(dataset_config['output'], os.W_OK):
        logger.error('Cannot write to output directory {d}'.format(
            d=dataset_config['output']))
        raise click.Abort()

    # Test existence of cache directory
    read_cache, write_cache = test_cache(dataset_config)

    logger.info('Job {i} / {n} - using config file {f}'.format(
                i=job_number, n=total_jobs, f=config))
    main(job_number, total_jobs, dataset_config, yatsm_config,
         check=check, resume=resume,
         do_not_run=do_not_run,
         read_cache=read_cache, write_cache=write_cache,
         validate_cache=check_cache)


# Runner
def run_line(line, X, images, image_IDs,
             dataset_config, yatsm_config,
             nrow, ncol, nband, dtype,
             do_not_run=False,
             read_cache=False, write_cache=False,
             validate_cache=False):
    """ Runs YATSM for a line

    Args:
      line (int): line to be run from image
      dates (ndarray): np.array of X feature from ordinal dates
      images (ndarray): np.array of image filenames
      image_IDs (iterable): list image identifying strings
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
      nrow (int): number of rows
      ncol (int): number of columns
      nband (int): number of bands
      dtype (type): NumPy datatype
      do_not_run (bool, optional): don't run YATSM
      read_cache (bool, optional): try to read from cache directory
        (default: False)
      write_cache (bool, optional): try to to write to cache directory
        (default: False)
      validate_cache (bool, optional): ensure data from cache file come from
        images specified in configuration (default: False)

    """
    # Setup output
    output = []

    Y = read_line(line, images, image_IDs, dataset_config,
                  ncol, nband, dtype,
                  read_cache=read_cache, write_cache=write_cache,
                  validate_cache=validate_cache)

    if do_not_run:
        return

    # About to run YATSM
    logger.debug('    running YATSM')
    for c in range(Y.shape[-1]):
        try:
            result = run_pixel(X, Y[..., c], dataset_config, yatsm_config,
                               px=c, py=line)
        except TSLengthException:
            continue

        output.extend(result)

    # Save output
    outfile = get_output_name(dataset_config, line)
    logger.debug('    saving YATSM output to {f}'.format(f=outfile))

    np.savez(outfile,
             version=__version__,
             record=np.array(output),
             **{k: v for k, v in yatsm_config.iteritems()})


def run_pixel(X, Y, dataset_config, yatsm_config, px=0, py=0):
    """ Run a single pixel through YATSM

    Args:
      X (ndarray): 2D (nimage x nband) feature input from ordinal date
      Y (ndarray): 2D (nband x nimage) image input
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
      px (int, optional):       X (column) pixel reference
      py (int, optional):       Y (row) pixel reference

    Returns:
      model_result (ndarray): NumPy array of model results from YATSM

    """
    # Extract design info
    design_info = X.design_info
    # Continue if valid observations are less than 50% of dataset
    valid = cyprep.get_valid_mask(
      Y[:dataset_config['mask_band'] - 1, :],
      dataset_config['min_values'],
      dataset_config['max_values']
    )
    if valid.sum() < Y.shape[1] / 2.0:
        raise TSLengthException('Not enough valid observations')

    # Otherwise continue with masked values
    valid = (valid * np.in1d(Y[dataset_config['mask_band'] - 1, :],
                             dataset_config['mask_values'],
                             invert=True)).astype(np.bool)

    Y = Y[:dataset_config['mask_band'] - 1, valid]
    X = X[valid, :]

    if yatsm_config['reverse']:
        # TODO: do this earlier
        X = np.flipud(X)
        Y = np.fliplr(Y)

    yatsm = YATSM(X, Y,
                  consecutive=yatsm_config['consecutive'],
                  threshold=yatsm_config['threshold'],
                  min_obs=yatsm_config['min_obs'],
                  min_rmse=yatsm_config['min_rmse'],
                  test_indices=yatsm_config['test_indices'],
                  retrain_time=yatsm_config['retrain_time'],
                  screening=yatsm_config['screening'],
                  screening_crit=yatsm_config['screening_crit'],
                  green_band=dataset_config['green_band'] - 1,
                  swir1_band=dataset_config['swir1_band'] - 1,
                  remove_noise=yatsm_config['remove_noise'],
                  dynamic_rmse=yatsm_config['dynamic_rmse'],
                  slope_test=yatsm_config['slope_test'],
                  lassocv=yatsm_config['lassocv'],
                  design_info=design_info,
                  px=px,
                  py=py,
                  logger=logger_algo)
    yatsm.run()

    if yatsm_config['commission_alpha']:
        yatsm.record = yatsm.commission_test(yatsm_config['commission_alpha'])

    if yatsm_config['robust']:
        yatsm.record = yatsm.robust_record

    if yatsm_config['calc_pheno']:
        ltm = pheno.LongTermMeanPhenology(
            yatsm,
            yatsm_config['red_index'], yatsm_config['nir_index'],
            yatsm_config['blue_index'], yatsm_config['scale'],
            yatsm_config['evi_index'], yatsm_config['evi_scale'])
        yatsm.record = ltm.fit(year_interval=yatsm_config['year_interval'],
                               q_min=yatsm_config['q_min'],
                               q_max=yatsm_config['q_max'])

    return yatsm.record


def main(job_number, total_jobs, dataset_config, yatsm_config,
         check=False, resume=False,
         do_not_run=False,
         read_cache=False, write_cache=False,
         validate_cache=False):
    """ Read in dataset and YATSM for a complete line

    Args:
      dataset_config (dict): dict of dataset configuration options
      yatsm_config (dict): dict of YATSM algorithm options
      check (bool, optional): check to make sure images are readible
      resume (bool, optional): do not overwrite existing results, instead
        continue from first non-existing result file
      do_not_run (bool, optional): Don't run YATSM
      read_cache (bool, optional): try to read from cache directory
        (default: False)
      write_cache (bool, optional): try to to write to cache directory
        (default: False)
      validate_cache (bool, optional): ensure data from cache file come from
        images specified in configuration (default: False)

    """
    # Read in dataset
    dataset = csvfile_to_dataset(
        dataset_config['input_file'],
        date_format=dataset_config['date_format']
    )
    dates = dataset['dates']
    sensors = dataset['sensors']
    images = dataset['images']

    image_IDs = get_image_IDs(images)

    # Check for existence of files and remove missing
    if check:
        to_delete = []
        for i, img in enumerate(images):
            if not os.path.isfile(img):
                logger.warning('Could not find file {f} -- removing'.
                               format(f=img))
                to_delete.append(i)

        if len(to_delete) == 0:
            logger.debug('Checked and found all input images')
        else:
            logger.warning('Removing {n} images'.format(n=len(to_delete)))
            dates = np.delete(dates, np.array(to_delete))
            images = np.delete(images, np.array(to_delete))

    # Get attributes of one of the images
    nrow, ncol, nband, dtype = get_image_attribute(images[0])

    # Calculate the lines this job ID works on
    job_lines = distribute_jobs(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Calculate X feature input
    X = patsy.dmatrix(yatsm_config['design_matrix'],
                      {'x': dates, 'sensor': sensors})

    # Start running YATSM
    start_time_all = time.time()
    logger.info('Starting to run lines')
    for job_line in job_lines:
        if resume:
            try:
                z = np.load(get_output_name(dataset_config, job_line))
            except:
                pass
            else:
                del z
                logger.debug('Already processed line {l}'.format(l=job_line))
                continue

        logger.debug('Running line {l}'.format(l=job_line))
        start_time = time.time()

        try:
            run_line(job_line, X, images, image_IDs,
                     dataset_config, yatsm_config,
                     nrow, ncol, nband, dtype,
                     do_not_run=do_not_run,
                     read_cache=read_cache, write_cache=write_cache,
                     validate_cache=validate_cache)
        except Exception as e:
            logger.error('Could not process line {l}'.format(l=job_line))
            logger.error(type(e))
            logger.error(str(e))

        logger.debug('Took {s}s to run'.format(
            s=round(time.time() - start_time, 2)))

    logger.info('Completed {n} lines in {m} minutes'.format(
        n=len(job_lines),
        m=round((time.time() - start_time_all) / 60.0, 2)
    ))
