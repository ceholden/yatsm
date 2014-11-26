try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import StringIO

import numpy as np


# CONFIG FILE PARSING
def parse_config_v0_1_x(config_file):
    """ Parses config file for version 0.1.x """
    # Defaults
    defaults = """
[dataset]
date_format=%Y%j
output_prefix = yatsm_r
use_bip_reader = false
cache_line_dir =
[YATSM]
retrain_time = 365.25
screening_crit = 400.0
robust = false
[classification]
training_image = None
mask_values = 0, 255
cache_xy =
    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.readfp(StringIO.StringIO(defaults))
    config.read(config_file)

    # Configuration for dataset
    dataset_config = {}

    dataset_config['input_file'] = config.get('dataset', 'input_file')
    dataset_config['date_format'] = config.get('dataset', 'date_format')
    dataset_config['output'] = config.get('dataset', 'output')
    dataset_config['output_prefix'] = config.get('dataset', 'output_prefix')
    dataset_config['n_bands'] = config.getint('dataset', 'n_bands')
    dataset_config['mask_band'] = config.getint('dataset', 'mask_band') - 1
    dataset_config['green_band'] = config.getint('dataset', 'green_band') - 1
    dataset_config['swir1_band'] = config.getint('dataset', 'swir1_band') - 1
    dataset_config['use_bip_reader'] = config.getboolean(
        'dataset', 'use_bip_reader')
    dataset_config['cache_line_dir'] = config.get('dataset', 'cache_line_dir')

    # Configuration for training and classification
    if config.has_section('classification'):
        dataset_config['training_image'] = config.get('classification',
                                                      'training_image')
        dataset_config['mask_values'] = config.get('classification',
                                                   'mask_values')
        if dataset_config['mask_values']:
            dataset_config['mask_values'] = np.array([
                int(v) for v in
                dataset_config['mask_values'].replace(' ', ',').split(',')
                if v != ','])
        dataset_config['cache_Xy'] = config.get('classification', 'cache_Xy')
        if not dataset_config['cache_Xy']:
            dataset_config['cache_Xy'] = None
        dataset_config['training_start'] = config.get('classification',
                                                      'training_start')
        dataset_config['training_end'] = config.get('classification',
                                                    'training_end')
        dataset_config['training_date_format'] = config.get(
            'classification', 'training_date_format')

    # Configuration for YATSM algorithm
    yatsm_config = {}

    yatsm_config['consecutive'] = config.getint('YATSM', 'consecutive')
    yatsm_config['threshold'] = config.getfloat('YATSM', 'threshold')
    yatsm_config['min_obs'] = config.getint('YATSM', 'min_obs')
    yatsm_config['min_rmse'] = config.getfloat('YATSM', 'min_rmse')
    yatsm_config['freq'] = config.get(
        'YATSM', 'freq').replace(',', ' ').split(' ')
    yatsm_config['freq'] = [int(v) for v in yatsm_config['freq']
                            if v != '']
    yatsm_config['test_indices'] = config.get(
        'YATSM', 'test_indices').replace(',', ' ').split(' ')
    yatsm_config['test_indices'] = np.array([int(b) for b in
                                             yatsm_config['test_indices']
                                             if b != ''])
    yatsm_config['retrain_time'] = config.getfloat('YATSM', 'retrain_time')
    yatsm_config['screening'] = config.get('YATSM', 'screening')
    yatsm_config['screening_crit'] = config.getfloat('YATSM', 'screening_crit')

    yatsm_config['lassocv'] = config.getboolean('YATSM', 'lassocv')
    yatsm_config['reverse'] = config.getboolean('YATSM', 'reverse')
    yatsm_config['robust'] = config.getboolean('YATSM', 'robust')

    return dataset_config, yatsm_config


def parse_config_v0_2_x(config_file):
    """ Parses config file for version 0.2.x """
    dataset_config, yatsm_config = parse_config_v0_1_x(config_file)

    defaults = """
[YATSM]
remove_noise = True
    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.readfp(StringIO.StringIO(defaults))
    config.read(config_file)

    yatsm_config['remove_noise'] = config.getboolean('YATSM', 'remove_noise')

    return dataset_config, yatsm_config


def parse_config_file(config_file):
    """ Parses config file into dictionary of attributes """

    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_config = None
    yatsm_config = None

    # Parse different versions
    version = config.get('metadata', 'version').split('.')

    # 0.1.x
    if version[0] == '0' and version[1] == '1':
        dataset_config, yatsm_config = parse_config_v0_1_x(config_file)
    if version[0] == '0' and version[1] == '2':
        dataset_config, yatsm_config = parse_config_v0_2_x(config_file)

    return (dataset_config, yatsm_config)
