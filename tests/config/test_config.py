""" Tests for yatsm.config
"""
import pytest
import yaml

from yatsm.config import parse_config
from yatsm.errors import InvalidConfigurationException


@pytest.mark.parametrize(('cfg', 'name'), [
    ({}, 'version'),
])
def test_parse_config_missing_required(tmpdir, cfg, name):
    fn = tmpdir.join('cfg.yml').strpath
    with open(fn, 'w') as fid:
        fid.write(yaml.dump(cfg))

    with pytest.raises(InvalidConfigurationException) as err:
        parse_config(fn)
    assert name in str(err) and 'is a required property' in str(err)
