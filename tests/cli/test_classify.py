""" Tests for ``yatsm classify``
"""
import os

from click.testing import CliRunner
import numpy as np

from yatsm.cli.main import cli


def test_classify_pass_1(example_timeseries, example_results, modify_config):
    """ Correctly run classification script
    """
    mod_cfg = {'dataset': {'output': example_results['results_dir']}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '-v', 'classify',
                cfg,
                example_results['example_classify_pickle'],
                '1', '1'
            ]
        )
        assert result.exit_code == 0
        # Try opening & check that classes are in the files
        for result in os.listdir(example_results['results_dir']):
            z = np.load(os.path.join(example_results['results_dir'], result))
            assert 'classes' in z
            assert 'class' in z['record'].dtype.names


def test_classify_pass_2(example_timeseries, example_results, modify_config):
    """ Correctly run classification script with --resume option
    """
    mod_cfg = {'dataset':
               {'output': example_results['results_dir_classified']}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '-v', 'classify', '--resume',
                cfg,
                example_results['example_classify_pickle'],
                '1', '1'
            ]
        )
        assert result.exit_code == 0
        # Try opening & check that classes are in the files
        for result in os.listdir(example_results['results_dir']):
            z = np.load(os.path.join(
                example_results['results_dir_classified'], result))
            assert 'classes' in z
            assert 'class' in z['record'].dtype.names
