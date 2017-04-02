""" Tests for yatsm.config
"""
import pytest
import yaml

from yatsm.config import (load_config,
                          parse_config,
                          validate_config)
from yatsm.errors import InvalidConfigurationException


@pytest.mark.parametrize(('cfg', 'name'), [
    ({}, 'version'),
])
def test_validate_config_missing_required(cfg, name):
    with pytest.raises(InvalidConfigurationException) as err:
        validate_config(cfg)
    assert name in str(err) and 'is a required property' in str(err)
