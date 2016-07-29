""" Tests for ``yatsm.classification``
"""
import pytest
import yaml

from yatsm import classification
from yatsm.errors import AlgorithmNotFoundException


@pytest.fixture(scope='function')
def write_yaml(tmpdir):
    """ Write ``dict`` to a temporary file and return filename """
    def _inner(d):
        f = tmpdir.mkdir('clf').join('cfg').strpath
        with open(f, 'w') as fid:
            yaml.dump(d, fid)
        return f
    return _inner


def test_cfg_to_algorithm_pass_1(write_yaml):
    """ Test ``yatsm.classification.cfg_to_algorithm`` with an empty config
    """
    cfg = {'algorithm': 'RandomForest'}
    classification.cfg_to_algorithm(write_yaml(cfg))


def test_cfg_to_algorithm_pass_2(write_yaml):
    """ Test ``yatsm.classification.cfg_to_algorithm`` with an empty config
    """
    cfg = {
        'algorithm': 'RandomForest',
        'RandomForest': {'init': {}, 'fit': {}}
    }
    classification.cfg_to_algorithm(write_yaml(cfg))


# FAILURES
def test_cfg_to_algorithm_fail_1(write_yaml):
    """ Fail because algorithm in config doesn't exist
    """
    cfg = {'algorithm': 'hopefully_not_an_algo'}
    with pytest.raises(AlgorithmNotFoundException):
        classification.cfg_to_algorithm(write_yaml(cfg))


def test_cfg_to_algorithm_fail_2(write_yaml):
    """ Fail because algorithm parameters don't exist
    """
    cfg = {
        'algorithm': 'RandomForest',
        'RandomForest': {'init': {'not_a_param': 42}, 'fit': {}}
    }
    with pytest.raises(TypeError):
        classification.cfg_to_algorithm(write_yaml(cfg))


def test_cfg_to_algorithm_fail_3(tmpdir):
    """ Fail because we don't use a YAML file
    """
    f = tmpdir.mkdir('clf').join('test').strpath
    with pytest.raises(IOError):
        classification.cfg_to_algorithm(f)
