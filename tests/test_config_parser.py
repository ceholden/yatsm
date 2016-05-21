""" Tests for yatsm.config_parser
"""
import os

import pytest

from yatsm import config_parser
from yatsm.regression.packaged import packaged_regressions


# YATSM: SECTION PARSING
@pytest.fixture(scope='function')
def YATSM_cfg(request):
    cfg = {
        'YATSM': {
            'prediction': packaged_regressions[0],
            'refit': {
                'prefix': [reg for reg in packaged_regressions],
                'prediction': [reg for reg in packaged_regressions]
            }
        }
    }
    return cfg


def test_parse_YATSM_config_1(YATSM_cfg):
    """ Test retrieval of packaged estimators
    """
    for estimator in packaged_regressions:
        YATSM_cfg['YATSM']['prediction'] = estimator
        config_parser._parse_YATSM_config(YATSM_cfg)


def test_parse_YATSM_config_2(YATSM_cfg):
    """ Test retrieval of packaged estimators that don't exist
    """
    with pytest.raises(KeyError):
        YATSM_cfg['YATSM']['prediction'] = 'not_an_estimator'
        config_parser._parse_YATSM_config(YATSM_cfg)


def test_parse_YATSM_config_3(YATSM_cfg):
    """ Test parsing of config without "refit" section
    """
    del YATSM_cfg['YATSM']['refit']
    cfg = config_parser._parse_YATSM_config(YATSM_cfg)
    assert 'refit' in cfg['YATSM']
    assert cfg['YATSM']['refit']['prefix'] == []
    assert cfg['YATSM']['refit']['prediction'] == []


def test_parse_YATSM_config_4(YATSM_cfg):
    """ Test parsing of config with "refit" estimators that don't exist
    """
    YATSM_cfg['YATSM']['refit']['prediction'] = 'not_an_estimator'
    with pytest.raises(KeyError):
        config_parser._parse_YATSM_config(YATSM_cfg)


def test_parse_YATSM_config_5(YATSM_cfg):
    """ Test parsing of config with misspecified "refit" section
    """
    YATSM_cfg['YATSM']['refit']['prefix'] = ['just_one_prefix']
    with pytest.raises(KeyError):
        config_parser._parse_YATSM_config(YATSM_cfg)


def test_parse_YATSM_config_6(YATSM_cfg):
    """ Test parsing of config with "stay_regularized" section
    """
    YATSM_cfg['YATSM']['refit']['stay_regularized'] = True
    config_parser._parse_YATSM_config(YATSM_cfg)


def test_parse_YATSM_config_7(YATSM_cfg):
    """ Test parsing of config with "stay_regularized" section
    """
    n = len(YATSM_cfg['YATSM']['refit']['prediction'])
    YATSM_cfg['YATSM']['refit']['stay_regularized'] = [True] * n
    config_parser._parse_YATSM_config(YATSM_cfg)
