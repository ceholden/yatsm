try:
    import ConfigParser as configparser
except ImportError:
    import configparser

import numpy as np


# CONFIG FILE PARSING
def _parse_config_v_zero_pt_one(config_file):
    """ Parses config file for version 0.1.x """
    # Defaults
    defaults = {
        'date_format': '%Y%j',
        'output_prefix': 'yatsm_r',
        'use_bip_reader': False,
        'training_image': None,
        'mask_values': '0, 255',
        'cache_Xy': False
    }

    config = configparser.ConfigParser(defaults=defaults, allow_no_value=True)
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

    # Configuration for training and classification
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
    dataset_config['training_start'] = config.get('classification',
                                                  'training_start')
    dataset_config['training_end'] = config.get('classification',
                                                'training_end')
    dataset_config['training_date_format'] = config.get('classification',
                                                        'training_date_format')

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
    yatsm_config['screening'] = config.get('YATSM', 'screening')
    yatsm_config['lassocv'] = config.getboolean('YATSM', 'lassocv')
    yatsm_config['reverse'] = config.getboolean('YATSM', 'reverse')
    yatsm_config['robust'] = config.getboolean('YATSM', 'robust')

    return (dataset_config, yatsm_config)


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
        dataset_config, yatsm_config = _parse_config_v_zero_pt_one(config_file)

    return (dataset_config, yatsm_config)
