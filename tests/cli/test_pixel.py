""" Test ``yatsm line``
"""
import os

from click.testing import CliRunner
import pytest

from yatsm.cli.main import cli


@pytest.mark.skipif("DISPLAY" not in os.environ, reason="requires display")
def test_cli_pixel_pass_1(example_timeseries):
    """ Correctly run for one pixel
    """
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'pixel',
         '--band', '5',
         '--plot', 'TS',
         '--style', 'ggplot',
         example_timeseries['config'], '1', '1'
         ])
    assert result.exit_code == 0
