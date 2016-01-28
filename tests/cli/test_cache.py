""" Tests for ``yatsm cache``
"""
from click.testing import CliRunner
import pytest

from yatsm.cli.main import cli


def test_cli_cache_pass_1(example_timeseries, example_results, modify_config,
                          tmpdir):
    """ Run correctly
    """
    mod_cfg = {'dataset': {'cache_line_dir': tmpdir.mkdir('cache').strpath}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(cli, [
            '-v', 'cache',
            cfg, '1', '1'
        ])

        assert result.exit_code == 0


def test_cli_cache_pass_2(example_timeseries, example_results, modify_config,
                          tmpdir):
    """ Run correctly, interlacing
    """
    mod_cfg = {'dataset': {'cache_line_dir': tmpdir.mkdir('cache').strpath}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(cli, [
            '-v', 'cache',
            '--interlace',
            cfg, '1', '1'
        ])
        assert result.exit_code == 0
