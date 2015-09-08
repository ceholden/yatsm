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
                         csvfile_to_dataframe)
from yatsm.reader import get_image_attribute, read_line
from yatsm.regression.transforms import harm
from yatsm.algorithms import ccdc, postprocess
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
         resume, check_cache, do_not_run, verbose_yatsm):
    if verbose_yatsm:
        logger_algo.setLevel(logging.DEBUG)

    # Parse config
    cfg = parse_config_file(config)

    if ('phenology' in cfg and 'calc_pheno' in cfg['phenology']) and not pheno:
        raise click.Abort('Could not import yatsm.phenology but phenology '
                          'metrics are requested')

    # Make sure output directory exists and is writable
    output_dir = cfg['dataset']['output']
    try:
        os.makedirs(output_dir)
    except OSError as e:
        # File exists
        if e.errno == 17:
            pass
        elif e.errno == 13:
            raise click.Abort('Cannot create output directory %s' % output_dir)

    if not os.access(output_dir, os.W_OK):
        raise click.Abort('Cannot write to output directory %s' % output_dir)

    # Test existence of cache directory
    read_cache, write_cache = test_cache(cfg['dataset'])

    logger.info('Job {i} of {n} - using config file {f}'.format(i=job_number,
                                                                n=total_jobs,
                                                                f=config))
    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              cfg['dataset']['date_format'])
    df['image_ID'] = get_image_IDs(df['filename'])

    # Get attributes of one of the images
    nrow, ncol, nband, dtype = get_image_attribute(df['filename'][0])

    # Calculate the lines this job ID works on
    job_lines = distribute_jobs(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Calculate X feature input
    kws = {'x': df['date']}
    kws.update(df.to_dict())
    X = patsy.dmatrix(cfg['YATSM']['design_matrix'], kws)

    # Form YATSM class arguments
    design_info = X.design_info
    fit_indices = np.arange(cfg['dataset']['n_bands'])
    if cfg['dataset']['mask_band'] is not None:
        fit_indices = fit_indices[:-1]

    if cfg['YATSM']['reverse']:
        X = np.flipud(X)

    # Begin process
    start_time_all = time.time()
    for line in job_lines:
        out = get_output_name(cfg['dataset'], line)

        if resume:
            try:
                np.load(out)
            except:
                pass
            else:
                logger.debug('Already processed line %s' % line)
                continue

        logger.debug('Running line %s' % line)
        start_time = time.time()

        Y = read_line(line, df['filename'], df['image_ID'], cfg['dataset'],
                      ncol, nband, dtype,
                      read_cache=read_cache, write_cache=write_cache,
                      validate_cache=False)
        if do_not_run:
            continue
        if cfg['YATSM']['reverse']:
            Y = np.fliplr(Y)

        output = []
        for col in np.arange(Y.shape[-1]):
            _Y = Y.take(col, axis=2)
            # Mask
            idx_mask = cfg['dataset']['mask_band'] - 1
            valid = cyprep.get_valid_mask(
                _Y,
                cfg['dataset']['min_values'],
                cfg['dataset']['max_values']).astype(bool)

            valid *= np.in1d(_Y.take(idx_mask, axis=0),
                             cfg['dataset']['mask_values'],
                             invert=True).astype(np.bool)

            _Y = np.delete(_Y, idx_mask, axis=0)[:, valid]
            _X = X[valid, :]

            # Run model
            algo = cfg['YATSM']['algorithm']
            yatsm = cfg['YATSM']['algorithm_cls'](
                fit_indices,
                design_info,
                lm=cfg['YATSM']['prediction_object'],
                px=col, py=line,
                **cfg[algo])
            yatsm.fit(_X, _Y)

            # Formulate save file metadata
            md = cfg[algo].copy()

            # Postprocess
            if cfg['YATSM']['commission_alpha']:
                yatsm.record = postprocess.commission_test(
                    yatsm, cfg['YATSM']['commission_alpha'])

            if cfg['YATSM']['robust']:
                yatsm.record = postprocess.robust_record(yatsm)

            if cfg['phenology']['enable']:
                ltm = pheno.LongTermMeanPhenology(yatsm, **cfg['phenology'])
                yatsm.record = ltm.fit(
                    year_interval=cfg['phenology']['year_interval'],
                    q_min=cfg['phenology']['q_min'],
                    q_max=cfg['phenology']['q_max'])
                md.update(cfg['YATSM']['phenology'])

            output.extend(yatsm.record)

        logger.debug('    Saving YATSM output to %s' % out)
        np.savez(out,
                 version=__version__,
                 record=np.array(output),
                 **{k: v for k, v in md.iteritems()})

        run_time = time.time() - start_time
        logger.debug('Line %s took %ss to run' % (line, run_time))

    logger.info('Completed {n} lines in {m} minutes'.format(
                n=len(job_lines),
                m=round((time.time() - start_time_all) / 60.0, 2)))
