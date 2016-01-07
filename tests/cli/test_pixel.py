""" Test ``yatsm line``
"""
import os

from click.testing import CliRunner
import matplotlib as mpl
import pytest

from yatsm.cli.main import cli

mpl_skip = pytest.mark.skipif(
    mpl.get_backend() != 'agg' and "DISPLAY" not in os.environ,
    reason='Requires either matplotlib "agg" backend or that DISPLAY" is set')


@mpl_skip
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
