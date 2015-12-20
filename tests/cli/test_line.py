""" Test yatsm line
"""
from click.testing import CliRunner

from yatsm.cli import line


def test_cli_line_pass_1(example_timeseries):
    """ Run correctly, with 1 of 1 jobs
    """
    runner = CliRunner()
    result = runner.invoke(line.line, [example_timeseries['config'], '1', '1'])
    assert result.exit_code == 0
