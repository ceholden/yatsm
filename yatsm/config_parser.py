try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import StringIO

import numpy as np

from log_yatsm import logger
from version import __version__

defaults = """
[metadata]
version = 0.3

[dataset]
input_file =
date_format = %Y%j
output =
output_prefix = yatsm_r
n_bands = 8
mask_band = 8
mask_values = 2, 3, 4, 255
valid_range = 0, 10000
green_band = 2
swir1_band = 5
use_bip_reader = true
cache_line_dir =

[YATSM]
consecutive = 5
threshold = 3
min_obs = 16
min_rmse = 150
design_matrix = 1 + x + harm(x, 1)
test_indices = 2, 4, 5
retrain_time = 365.25
screening = RLM
screening_crit = 400.0
remove_noise = True
dynamic_rmse = False
lassocv = False
reverse = False
robust = False
commission_alpha =

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


# CONFIG FILE PARSING
def parse_dataset_config(config):
    """ Parses config for dataset section """
    # Configuration for dataset
    dataset_config = {}

    dataset_config['input_file'] = config.get('dataset', 'input_file')
    dataset_config['date_format'] = config.get('dataset', 'date_format')
    dataset_config['output'] = config.get('dataset', 'output')
    dataset_config['output_prefix'] = config.get('dataset', 'output_prefix')
    dataset_config['n_bands'] = config.getint('dataset', 'n_bands')
    dataset_config['mask_band'] = config.getint('dataset', 'mask_band')

    mask_values = config.get('dataset', 'mask_values')
    dataset_config['mask_values'] = [
        int(mv) for mv in mask_values.replace(',', ' ').split(' ') if mv != '']

    valid_range = config.get('dataset', 'valid_range').replace(
        ' ', ',').split(';')
    min_values, max_values = [], []
    for _vr in valid_range:
        if _vr:
            minmax = [int(_v) for _v in _vr.split(',') if _v != '']
            min_values.append(minmax[0])
            max_values.append(minmax[1])
    if len(min_values) != len(max_values):
        raise ValueError('Could not parse minimum and maximum range values')
    if len(min_values) == 1:
        # Extend min/max values to match number of bands if not
        min_values = min_values * dataset_config['n_bands']
        max_values = max_values * dataset_config['n_bands']
    elif len(min_values) != dataset_config['n_bands'] - 1:
        raise ValueError('Valid range of data must be specified for every '
                         'dataset band separately, or specified just once for'
                         ' all bands')
    dataset_config['min_values'] = np.array(min_values, dtype=np.int32)
    dataset_config['max_values'] = np.array(max_values, dtype=np.int32)

    dataset_config['green_band'] = config.getint('dataset', 'green_band')
    dataset_config['swir1_band'] = config.getint('dataset', 'swir1_band')
    dataset_config['use_bip_reader'] = config.getboolean(
        'dataset', 'use_bip_reader')
    dataset_config['cache_line_dir'] = config.get('dataset', 'cache_line_dir')

    return dataset_config


def parse_algorithm_config(config):
    # Configuration for YATSM algorithm
    yatsm_config = {}

    yatsm_config['consecutive'] = config.getint('YATSM', 'consecutive')
    yatsm_config['threshold'] = config.getfloat('YATSM', 'threshold')
    yatsm_config['min_obs'] = config.getint('YATSM', 'min_obs')
    yatsm_config['min_rmse'] = config.getfloat('YATSM', 'min_rmse')
    yatsm_config['design_matrix'] = config.get('YATSM', 'design_matrix')
    yatsm_config['test_indices'] = config.get(
        'YATSM', 'test_indices').replace(',', ' ').split(' ')
    yatsm_config['test_indices'] = np.array([
        int(b) for b in yatsm_config['test_indices'] if b != ''])
    yatsm_config['retrain_time'] = config.getfloat('YATSM', 'retrain_time')
    yatsm_config['screening'] = config.get('YATSM', 'screening')
    yatsm_config['screening_crit'] = config.getfloat('YATSM', 'screening_crit')
    yatsm_config['remove_noise'] = config.getboolean('YATSM', 'remove_noise')
    yatsm_config['dynamic_rmse'] = config.getboolean('YATSM', 'dynamic_rmse')
    yatsm_config['lassocv'] = config.getboolean('YATSM', 'lassocv')
    yatsm_config['reverse'] = config.getboolean('YATSM', 'reverse')
    yatsm_config['robust'] = config.getboolean('YATSM', 'robust')
    commission_alpha = config.get('YATSM', 'commission_alpha')
    yatsm_config['commission_alpha'] = (float(commission_alpha) if
                                        commission_alpha else None)

    return yatsm_config


def parse_classification_config(config):
    # Configuration for training and classification
    dataset_config = {}

    if config.has_section('classification'):
        dataset_config['training_image'] = config.get(
            'classification', 'training_image')
        dataset_config['roi_mask_values'] = config.get(
            'classification', 'roi_mask_values')
        if dataset_config['roi_mask_values']:
            dataset_config['roi_mask_values'] = np.array([
                int(v) for v in
                dataset_config['roi_mask_values'].replace(' ', ',').split(',')
                if v != ''
            ])
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

    return dataset_config


def parse_phenology_config(config):
    """ Parse phenology config """
    yatsm_config = {}

    yatsm_config['calc_pheno'] = config.getboolean('phenology', 'calc_pheno')
    if not yatsm_config['calc_pheno']:
        return yatsm_config

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

    return yatsm_config


def parse_config_file(config_file):
    """ Parses config file into dictionary of attributes """

    config = configparser.ConfigParser(allow_no_value=True)
    config.readfp(StringIO.StringIO(defaults))
    config.read(config_file)

    version = config.get('metadata', 'version')

    # Warn on difference in minor or major version
    mm_config_version = version.split('.')[0:2]
    mm_yatsm_version = __version__.split('.')[0:2]
    if mm_config_version[0] != mm_yatsm_version[0] or \
            mm_config_version[1] != mm_yatsm_version[1]:
        logger.warning('Config file version does not match YATSM version')
        logger.warning('    config file: v{v}'.format(v=version))
        logger.warning('    YATSM: v{v}'.format(v=__version__))

    dataset_config = {}
    yatsm_config = {}

    dataset_config.update(parse_dataset_config(config))
    yatsm_config.update(parse_algorithm_config(config))
    yatsm_config.update(parse_phenology_config(config))
    dataset_config.update(parse_classification_config(config))

    return (dataset_config, yatsm_config)
