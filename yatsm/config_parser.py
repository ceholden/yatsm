try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import logging
import StringIO

import numpy as np

from version import __version__

FORMAT = '%(asctime)s:%(levelname)s:%(module)s.%(funcName)s:%(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger('yatsm')


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
cache_training =
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
        dataset_config['cache_training'] = config.get(
            'classification', 'cache_training')
        if not dataset_config['cache_training']:
            dataset_config['cache_training'] = None
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
[dataset]
mask_values = 2, 3, 4, 255
[YATSM]
remove_noise = True
dynamic_rmse = False
commission_alpha =
    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.readfp(StringIO.StringIO(defaults))
    config.read(config_file)

    mask_values = config.get('dataset', 'mask_values')
    dataset_config['mask_values'] = [
        int(mv) for mv in mask_values.replace(',', ' ').split(' ') if mv != '']

    yatsm_config['remove_noise'] = config.getboolean('YATSM', 'remove_noise')
    yatsm_config['dynamic_rmse'] = config.getboolean('YATSM', 'dynamic_rmse')
    commission_alpha = config.get('YATSM', 'commission_alpha')
    yatsm_config['commission_alpha'] = (float(commission_alpha) if
                                        commission_alpha else None)

    return dataset_config, yatsm_config


def parse_config_v0_3_x(config_file):
    """ Parse config file for version 0.3.x """
    dataset_config, yatsm_config = parse_config_v0_2_x(config_file)

    defaults = """
[phenology]
calc_pheno = False
red_index = 2
nir_index = 3
blue_index = 0
scale = 0.0001
evi_index =
evi_scale =
year_interval = 3
q_min = 10
q_max = 90
    """

    config = configparser.ConfigParser(allow_no_value=True)
    config.readfp(StringIO.StringIO(defaults))
    config.read(config_file)

    yatsm_config['calc_pheno'] = config.getboolean('phenology', 'calc_pheno')
    if not yatsm_config['calc_pheno']:
        return dataset_config, yatsm_config

    red_index = config.get('phenology', 'red_index')
    yatsm_config['red_index'] = int(red_index) if red_index else None
    nir_index = config.get('phenology', 'nir_index')
    yatsm_config['nir_index'] = int(nir_index) if nir_index else None
    blue_index = config.get('phenology', 'blue_index')
    yatsm_config['blue_index'] = int(blue_index) if blue_index else None
    scale = config.get('phenology', 'scale')
    yatsm_config['scale'] = float(scale) if scale else None
    evi_index = config.get('phenology', 'evi_index')
    yatsm_config['evi_index'] = int(evi_index) if evi_index else None
    evi_scale = config.get('phenology', 'evi_scale')
    yatsm_config['evi_scale'] = float(evi_scale) if evi_scale else None
    year_int = config.get('phenology', 'year_interval')
    yatsm_config['year_interval'] = int(year_int) if year_int else None
    q_min = config.get('phenology', 'q_min')
    yatsm_config['q_min'] = float(q_min) if q_min else None
    q_max = config.get('phenology', 'q_max')
    yatsm_config['q_max'] = float(q_max) if q_max else None

    return dataset_config, yatsm_config


def parse_config_file(config_file):
    """ Parses config file into dictionary of attributes """

    config = configparser.ConfigParser()
    config.read(config_file)

    dataset_config = None
    yatsm_config = None

    version = config.get('metadata', 'version')

    # Warn on difference in minor or major version
    mm_config_version = version.split('.')[0:2]
    mm_yatsm_version = __version__.split('.')[0:2]
    if mm_config_version[0] != mm_yatsm_version[0] or \
            mm_config_version[1] != mm_yatsm_version[1]:
        logger.warning('Config file version does not match YATSM version')
        logger.warning('    config file: v{v}'.format(v=version))
        logger.warning('    YATSM: v{v}'.format(v=__version__))

    dataset_config, yatsm_config = parse_config_v0_3_x(config_file)

    return (dataset_config, yatsm_config)
