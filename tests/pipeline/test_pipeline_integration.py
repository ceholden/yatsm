""" Integration tests running pipelines
"""
import pytest

from yatsm.api import Config
from yatsm.pipeline import Pipeline, Task, Pipe


def test_pipeline_run_config1_1(config1):
    api = Config(config1)

    pipe = Pipe()
    pl = api.get_pipeline(pipe)
    pipe = pl.run(pipe)

    assert 'ndvi' in pipe.data
    assert 'ndmi' in pipe.data


def test_pipeline_run_config1_2(configfile1):
    api = Config.from_file(configfile1)

    pipe = Pipe()
    pl = api.get_pipeline(pipe)
    pipe = pl.run(pipe)

    assert 'ndvi' in pipe.data
    assert 'ndmi' in pipe.data
