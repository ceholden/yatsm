""" Test ``yatsm train``
"""
import os

from click.testing import CliRunner
import matplotlib as mpl
import pytest

from yatsm.cli.main import cli

mpl_skip = pytest.mark.skipif(
    mpl.get_backend() != 'agg' and "DISPLAY" not in os.environ,
    reason='Requires either matplotlib "agg" backend or that DISPLAY" is set')
xfail = pytest.mark.xfail(reason='Will fail until v0.7.0 stabilizes')


@xfail
def test_train_pass_1(example_timeseries, example_results, modify_config,
                      tmpdir):
    """ Correctly run training script
    """
    mod_cfg = {'dataset': {'output': example_results['results_dir']}}
    tmppkl = tmpdir.join('tmp.pkl').strpath
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '-v', 'train',
                cfg,
                example_results['classify_config'],
                tmppkl
            ]
        )
    assert result.exit_code == 0


@xfail
def test_train_pass_2(example_timeseries, example_results, modify_config,
                      tmpdir):
    """ Correctly run training script, overwriting a result
    """
    mod_cfg = {'dataset': {'output': example_results['results_dir']}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '-v', 'train', '--overwrite',
                cfg,
                example_results['classify_config'],
                example_results['example_classify_pickle']
            ]
        )
    assert result.exit_code == 0


@mpl_skip
@xfail
def test_train_pass_3(example_timeseries, example_results, modify_config):
    """ Correctly run training script with plots
    """
    mod_cfg = {'dataset': {'output': example_results['results_dir']}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '-v', 'train', '--overwrite',
                '--plot', '--diagnostics',
                cfg,
                example_results['classify_config'],
                example_results['example_classify_pickle']
            ]
        )
        assert result.exit_code == 0


# FAILURES
@xfail
def test_train_fail_1(example_timeseries, example_results, modify_config,
                      tmpdir):
    """ Fail because of existing pickle file
    """
    mod_cfg = {'dataset': {'output': example_results['results_dir']}}
    with modify_config(example_timeseries['config'], mod_cfg) as cfg:
        runner = CliRunner()
        result = runner.invoke(
            cli, [
                '-v', 'train',
                cfg,
                example_results['classify_config'],
                example_results['example_classify_pickle']
            ]
        )
    assert result.exit_code == 1
    assert '<model> exists and --overwrite was not specified' in result.output
