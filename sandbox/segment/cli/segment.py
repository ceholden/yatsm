""" Command line interface for running YATSM on image segments

Notes:
    1. Probably need to use NumPy masked arrays for summary to segment scale
    2. Should I ravel the row and columns?

"""
from datetime import datetime as dt
import logging
import sys

import click
import numpy as np
import patsy

import yatsm.cache
from yatsm.cli import options
import yatsm.config_parser
import yatsm._cyprep as cyprep
import yatsm.reader
import yatsm.segment
import yatsm.utils
import yatsm.yatsm

from yatsm.regression.transforms import harm

logger = logging.getLogger('yatsm')


def _temp_plot(dates, Y_seg_mean, Y_seg_std, Y_seg_stderr, Y_seg_mask,
               seg_id, plot_idx, results=None):
    import matplotlib.pyplot as plt

    seg_id -= 1
    plot_idx = 5
    plt.subplot(3, 1, 1)
    plt.plot(dates[Y_seg_mask[seg_id, :]],
             Y_seg_mean[seg_id, plot_idx, Y_seg_mask[seg_id, :]], 'ro')
    plt.ylabel('Mean idx {i}'.format(i=plot_idx))

    plt.subplot(3, 1, 2)
    plt.plot(dates[Y_seg_mask[seg_id, :]],
             Y_seg_std[seg_id, plot_idx, Y_seg_mask[seg_id, :]], 'ro')
    plt.ylabel('Std idx {i}'.format(i=plot_idx))

    plt.subplot(3, 1, 3)
    plt.errorbar(dates[Y_seg_mask[seg_id, :]],
                 Y_seg_mean[seg_id, plot_idx, Y_seg_mask[seg_id, :]],
                 yerr=Y_seg_stderr[seg_id, plot_idx, Y_seg_mask[seg_id, :]],
                 fmt='o')
    plt.ylabel('Mean/stderr idx {i}'.format(i=plot_idx))

    if results is not None:
        for i, r in enumerate(results.record):
            mx = np.arange(r['start'], r['end'], 1)
            from IPython.core.debugger import Pdb
            Pdb().set_trace()
            mX = patsy.dmatrix(results.design_info, {'x': mx})
            my = np.dot(r['coef'][:, plot_idx], mX)

            # dates =

    plt.show()


def read_data(cfg, lines, ravel=True):
    """ Read multiple lines of a timeseries into NumPy array

    Args:
      cfg (dict): YATSM dataset configuration
      lines (iterable): sequence of lines to read from timeseries stack
      ravel (bool, optional): ravel row and column into one dimension

    Returns:
      np.ndarray: lines of timeseries data

    """
    # Test existence of cache directory
    read_cache, write_cache = yatsm.cache.test_cache(cfg)

    # Find dataset and read in
    dataset = yatsm.utils.csvfile_to_dataframe(cfg['input_file'],
                                               date_format=cfg['date_format'])
    dates = dataset['dates']
    sensors = dataset['sensors']
    images = dataset['images']

    image_IDs = yatsm.utils.get_image_IDs(images)
    nrow, ncol, nband, dtype = yatsm.reader.get_image_attribute(images[0])

    if ravel:
        Y = np.empty((len(lines) * ncol, nband, len(images)), dtype=dtype)
        for i, line in enumerate(lines):
            _line = yatsm.reader.read_line(
                line, images, image_IDs, cfg, ncol, nband, dtype,
                read_cache=read_cache, write_cache=write_cache,
                validate_cache=True)
            Y[i * ncol:(i + 1) * ncol, ...] = np.rollaxis(_line, 2)
    else:
        Y = np.empty((len(lines), nband, len(images), ncol), dtype=dtype)
        for i, line in enumerate(lines):
            Y[i, ...] = yatsm.reader.read_line(
                line, images, image_IDs, cfg, ncol, nband, dtype,
                read_cache=read_cache, write_cache=write_cache,
                validate_cache=True)

    return Y, dates


@click.command(short_help='Run YATSM on a segmented image')
@options.arg_config_file
@options.arg_job_number
@options.arg_total_jobs
@click.argument('seg_id', type=click.INT, metavar='<seg_id>')
@click.pass_context
def segment(ctx, config, job_number, total_jobs, seg_id):
    # Parse config
    dataset_config, yatsm_config = \
        yatsm.config_parser.parse_config_file(config)

    # Read in segmentation image
    if not yatsm_config['segmentation']:
        logger.error('No segmentation image specified in configuration file.')
        sys.exit(1)
    segment = yatsm.reader.read_image(yatsm_config['segmentation'])[0]

    # Calculate segments for this job
    n_segment = segment.max()
    job_segments = yatsm.utils.distribute_jobs(job_number, total_jobs,
                                               n_segment, interlaced=False)
    job_segments += segment.min()

    # What lines are required?
    job_lines = yatsm.segment.segments_to_lines(segment, job_segments)

    # Read and store all required lines
    Y, ord_dates = read_data(dataset_config, job_lines, ravel=True)
    dates = np.array([dt.fromordinal(d) for d in ord_dates])

    # Create design matrix
    X = patsy.dmatrix(yatsm_config['design_matrix'],
                      {'x': ord_dates})

    # Preprocess timeseries for each segment
    Y_mask = np.empty((Y.shape[0], Y.shape[2]), dtype=np.bool)
    # TODO: I'm sure there's a more efficient way...
    for pix in range(Y.shape[0]):
        Y_mask[pix, :] = ~cyprep.get_valid_mask(
            Y[pix, :dataset_config['mask_band'] - 1, :,],
            dataset_config['min_values'], dataset_config['max_values'])

    # Apply Fmask
    Y_mask *= np.in1d(Y[:, dataset_config['mask_band'] - 1, :],
                      dataset_config['mask_values']).reshape(Y_mask.shape)

    # Mask Y
    Y_mask = np.ones((Y.shape[0], Y.shape[1] - 1, Y.shape[2]), np.bool) \
        * Y_mask[:, np.newaxis, :]
    Y = np.ma.masked_array(Y[:, :dataset_config['mask_band'] - 1, :], Y_mask)

    # Preprocess segments
    Y_seg_n = np.ones((len(job_segments), Y.shape[2]), np.int16)
    Y_seg_mask = np.ones((len(job_segments), Y.shape[2]), np.bool)

    Y_seg_mean = np.empty((len(job_segments), Y.shape[1], Y.shape[2]))
    Y_seg_var = np.empty((len(job_segments), Y.shape[1], Y.shape[2]))
    Y_seg_std = np.empty((len(job_segments), Y.shape[1], Y.shape[2]))
    Y_seg_stderr = np.empty((len(job_segments), Y.shape[1], Y.shape[2]))

    for i, region_ID in enumerate(job_segments):
        reg_row, reg_col = np.where(segment == region_ID)
        reg_idx = np.ravel_multi_index((reg_row, reg_col), segment.shape)
        reg_Y = Y[reg_idx, :, :]

        Y_seg_n[i, :] = reg_Y[:, 0, :].mask.sum(axis=0)
        Y_seg_mask[i, :] = Y_seg_n[i, :] != reg_Y.shape[0]

        # Summary stats
        Y_seg_mean[i, :, :] = np.ma.mean(reg_Y, axis=0).data
        Y_seg_var[i, :, :] = np.ma.var(reg_Y, axis=0).data
        Y_seg_std[i, :, :] = np.ma.std(reg_Y, axis=0).data
        Y_seg_stderr[i, :, :] = np.ma.std(reg_Y, axis=0).data / np.sqrt(Y_seg_n[i, :])

    plot_idx = 4
    _temp_plot(dates, Y_seg_mean, Y_seg_std, Y_seg_stderr, Y_seg_mask,
               seg_id, plot_idx)

    from IPython.core.debugger import Pdb
    Pdb().set_trace()

    _yatsm = yatsm.yatsm.YATSM(
        X[Y_seg_mask[seg_id, :]],
        Y_seg_mean[seg_id, :, :],
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
        design_info=X.design_info,
        lassocv=yatsm_config['lassocv'],
        px=seg_id,
        py=0,
        logger=logger)
    _yatsm.run()

    breakpoints = _yatsm.record['break'][_yatsm.record['break'] != 0]

    print('Found {n} breakpoints'.format(n=breakpoints.size))
    if breakpoints.size > 0:
        for i, bp in enumerate(breakpoints):
            print('Break {0}: {1}'.format(
                i, dt.fromordinal(bp).strftime('%Y-%m-%d')))

    _temp_plot(dates, Y_seg_mean, Y_seg_std, Y_seg_stderr, Y_seg_mask,
               seg_id, plot_idx, results=_yatsm)

if __name__ == '__main__':
    segment()
