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
from yatsm.algorithms import postprocess
try:
    import yatsm.phenology.longtermmean as pheno
except ImportError as e:
    pheno = None
    pheno_exception = e.message
from yatsm.version import __version__

logger = logging.getLogger('yatsm')


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
@click.pass_context
def line(ctx, config, job_number, total_jobs,
         resume, check_cache, do_not_run):
    # Parse config
    cfg = parse_config_file(config)

    if ('phenology' in cfg and cfg['phenology'].get('enable')) and not pheno:
        click.secho('Could not import yatsm.phenology but phenology metrics '
                    'are requested', fg='red')
        click.secho('Error: %s' % pheno_exception, fg='red')
        raise click.Abort()

    # Make sure output directory exists and is writable
    output_dir = cfg['dataset']['output']
    try:
        os.makedirs(output_dir)
    except OSError as e:
        # File exists
        if e.errno == 17:
            pass
        elif e.errno == 13:
            click.secho('Cannot create output directory %s' % output_dir,
                        fg='red')
            raise click.Abort()

    if not os.access(output_dir, os.W_OK):
        click.secho('Cannot write to output directory %s' % output_dir,
                    fg='red')
        raise click.Abort()

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
    if nband != cfg['dataset']['n_bands']:
        logger.error('Number of bands in image %s (%i) do not match number '
                     'in configuration file (%i)' %
                     (df['filename'][0], nband, cfg['dataset']['n_bands']))
        raise click.Abort()

    # Calculate the lines this job ID works on
    job_lines = distribute_jobs(job_number, total_jobs, nrow)
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Calculate X feature input
    df['x'] = df['date']
    X = patsy.dmatrix(cfg['YATSM']['design_matrix'], data=df)
    cfg['YATSM']['design'] = X.design_info.column_name_indexes

    if cfg['YATSM']['reverse']:
        X = np.flipud(X)

    # Create output metadata to save
    md = {
        'YATSM': cfg['YATSM'],
        cfg['YATSM']['algorithm']: cfg[cfg['YATSM']['algorithm']]
    }
    if cfg['phenology']['enable']:
        md.update({'phenology': cfg['phenology']})

    # Initialize timeseries model
    cls = cfg['YATSM']['algorithm_cls']
    algo_cfg = cfg[cfg['YATSM']['algorithm']]
    yatsm = cls(estimator=cfg['YATSM']['prediction_object'],
                **algo_cfg.get('init', {}))

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
            _dates = df['date'].values[valid]

            # Run model
            yatsm.px = col
            yatsm.py = line

            try:
                yatsm.fit(_X, _Y, _dates, **algo_cfg.get('fit', {}))
            except TSLengthException:
                continue

            if yatsm.record is None or len(yatsm.record) == 0:
                continue

            # Postprocess
            if cfg['YATSM'].get('commission_alpha'):
                yatsm.record = postprocess.commission_test(
                    yatsm, cfg['YATSM']['commission_alpha'])

            for prefix, estimator in zip(
                    cfg['YATSM']['refit']['prefix'],
                    cfg['YATSM']['refit']['prediction_object']):
                yatsm.record = postprocess.refit_record(
                    yatsm, prefix, estimator, keep_regularized=True)

            if cfg['phenology']['enable']:
                pcfg = cfg['phenology']
                ltm = pheno.LongTermMeanPhenology(**pcfg.get('init', {}))
                yatsm.record = ltm.fit(yatsm, **pcfg.get('fit', {}))

            output.extend(yatsm.record)

        logger.debug('    Saving YATSM output to %s' % out)
        np.savez(out,
                 record=np.array(output),
                 version=__version__,
                 metadata=md)

        run_time = time.time() - start_time
        logger.debug('Line %s took %ss to run' % (line, run_time))

    logger.info('Completed {n} lines in {m} minutes'.format(
                n=len(job_lines),
                m=round((time.time() - start_time_all) / 60.0, 2)))
