""" Command line interface for running YATSM on image lines """
import logging
import os
import time

import click
import numpy as np

from . import options
from ..cache import test_cache
from ..config_parser import parse_config_file
from ..errors import TSLengthException
from ..io import get_image_attribute, mkdir_p, read_line
from ..utils import (distribute_jobs, get_output_name, get_image_IDs,
                     csvfile_to_dataframe, copy_dict_filter_key)
from ..algorithms import postprocess
try:
    from ..phenology import longtermmean as pheno
except ImportError as e:
    pheno = None
    pheno_exception = e.message
from ..version import __version__

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

    if ('phenology' in cfg and cfg['phenology'].get('enable')):
        if not pheno:
            raise click.ClickException('Could not import yatsm.phenology but '
                                       'phenology metrics are requested (%s)' %
                                       pheno_exception)

    logger.info('Job {i} of {n} - using config file {f}'.format(
        i=job_number, n=total_jobs, f=config))

    # Make sure output directory exists and is writable
    output_dir = cfg['dataset']['output']
    try:
        mkdir_p(output_dir)
    except OSError as err:
        raise click.ClickException('Cannot create output directory %s (%s)' %
                                   (output_dir, str(err)))
    if not os.access(output_dir, os.W_OK):
        raise click.ClickException('Cannot write to output directory %s' %
                                   output_dir)

    # Test existence of cache directory
    read_cache, write_cache = test_cache(cfg['dataset'])

    # Dataset information
    df = csvfile_to_dataframe(cfg['dataset']['input_file'],
                              cfg['dataset']['date_format'])
    df['image_ID'] = get_image_IDs(df['filename'])
    df['x'] = df['date']
    dates = df['date'].values

    # Get attributes of one of the images
    nrow, ncol, nband, dtype = get_image_attribute(df['filename'][0])
    if nband != cfg['dataset']['n_bands']:
        raise click.ClickException(
            'Number of bands in image %s (%i) do not match number '
            'in configuration file (%i)' %
            (df['filename'][0], nband, cfg['dataset']['n_bands']))

    # Calculate the lines this job ID works on
    try:
        job_lines = distribute_jobs(job_number, total_jobs, nrow)
    except ValueError as err:
        raise click.ClickException(str(err))
    logger.debug('Responsible for lines: {l}'.format(l=job_lines))

    # Initialize timeseries model
    model = cfg['YATSM']['algorithm_cls']
    algo_cfg = cfg[cfg['YATSM']['algorithm']]
    yatsm = model(estimator=cfg['YATSM']['estimator'],
                  **algo_cfg.get('init', {}))

    # Setup algorithm and create design matrix (if needed)
    X = yatsm.setup(df, **cfg)
    if hasattr(X, 'design_info'):
        cfg['YATSM']['design'] = X.design_info.column_name_indexes
    else:
        cfg['YATSM']['design'] = {}

    # Flip for reverse
    if cfg['YATSM']['reverse']:
        X = np.flipud(X)

    # Create output metadata to save
    algo = cfg['YATSM']['algorithm']
    md = {
        # Do not copy over prediction objects
        # Pickled objects potentially unstable across library versions
        'YATSM': copy_dict_filter_key(cfg['YATSM'], '.*object.*'),
        algo: cfg[algo].copy()
    }
    if cfg['phenology']['enable']:
        md.update({'phenology': cfg['phenology']})

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
            # Preprocess
            _X, _Y, _dates = yatsm.preprocess(X, _Y, dates, **cfg['dataset'])

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

            for prefix, estimator, stay_reg, fitopt in zip(
                    cfg['YATSM']['refit']['prefix'],
                    cfg['YATSM']['refit']['prediction_object'],
                    cfg['YATSM']['refit']['stay_regularized'],
                    cfg['YATSM']['refit']['fit']):
                yatsm.record = postprocess.refit_record(
                    yatsm, prefix, estimator,
                    fitopt=fitopt, keep_regularized=stay_reg)

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
