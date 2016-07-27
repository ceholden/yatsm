""" Test ``yatsm line``
"""
import os

from click.testing import CliRunner
import pytest

# from yatsm.cli import line


# PASSES
# def test_cli_line_pass_1(example_timeseries):
#     """ Run correctly, with 1 of 5 jobs
#     """
#     runner = CliRunner()
#     result = runner.invoke(line.line, [example_timeseries['config'], '1', '5'],
#                            catch_exceptions=False)
#     assert result.exit_code == 0
#
#
# def test_cli_line_pass_2(example_timeseries):
#     """ Run correctly, with --resume
#     """
#     runner = CliRunner()
#     result = runner.invoke(
#         line.line,
#         ['--resume', example_timeseries['config'], '1', '5'],
#         catch_exceptions=False)
#     assert result.exit_code == 0
#
#
# def test_cli_line_pass_3(example_timeseries):
#     """ Run correctly, with --do-not-run
#     """
#     runner = CliRunner()
#     result = runner.invoke(
#         line.line,
#         ['--do-not-run', example_timeseries['config'], '1', '5'],
#         catch_exceptions=False)
#     assert result.exit_code == 0
#
#
# def test_cli_line_pass_commission(example_timeseries, modify_config):
#     """ Run correctly, with commission test
#     """
#     with modify_config(example_timeseries['config'],
#                        {'YATSM': {'commission_alpha': 0.10}}) as cfg:
#         runner = CliRunner()
#         result = runner.invoke(line.line,
#                                [cfg, '1', '5'],
#                                catch_exceptions=False)
#         assert result.exit_code == 0
#
#
# def test_cli_line_pass_reverse(example_timeseries, modify_config):
#     """ Run correctly, with commission test
#     """
#     with modify_config(example_timeseries['config'],
#                        {'YATSM': {'reverse': True}}) as cfg:
#         runner = CliRunner()
#         result = runner.invoke(line.line,
#                                [cfg, '1', '5'],
#                                catch_exceptions=False)
#         assert result.exit_code == 0
#
#
# # Users opt-in to using numba and ignore by not installing
# # Seems like this scenario -- installed pickles using JIT but run disabled
# # using CLI -- is very unlikely
# # This behavior seems to have started between 0.24 - 0.25 of numba
# @pytest.mark.skipif(
#     'NUMBA_DISABLE_JIT' in os.environ,
#     reason="Numba disabled, but would use pickle that is JIT-d in test"
# )
# def test_cli_line_pass_refit_rlm(example_timeseries, modify_config):
#     """ Run correctly, with commission test
#     """
#     refit = {
#         'prefix': ['robust'],
#         'prediction': ['rlm_maxiter10'],  # use a pre-packaged
#         'stay_regularized': True
#     }
#     with modify_config(example_timeseries['config'],
#                        {'YATSM': {'refit': refit}}) as cfg:
#         runner = CliRunner()
#         result = runner.invoke(line.line,
#                                [cfg, '1', '5'],
#                                catch_exceptions=False)
#         assert result.exit_code == 0
#
#
# # FAILURES
# def test_cli_line_fail_1(example_timeseries):
#     """ Run correctly, but fail with 6 of 5 jobs
#     """
#     runner = CliRunner()
#     result = runner.invoke(line.line, [example_timeseries['config'], '6', '5'],
#                            catch_exceptions=False)
#     assert result.exit_code == 1
#     assert 'No jobs assigned' in result.output
#
#
# def test_cli_line_fail_2(example_timeseries, modify_config):
#     """ Mis-specify the number of bands in dataset
#     """
#     with modify_config(example_timeseries['config'],
#                        {'dataset': {'n_bands': 100}}) as cfg:
#         runner = CliRunner()
#         result = runner.invoke(line.line,
#                                [cfg, '1', '5'],
#                                catch_exceptions=False)
#         assert result.exit_code == 1
#         assert 'Number of bands in' in result.output
#
#
# # PHENOLOGY
# @pytest.fixture(scope='function')
# def break_pheno(request):
#     def fix_pheno():
#         line.pheno = pheno_status
#     pheno_status = line.pheno
#     line.pheno = None
#     line.pheno_exception = "Don't panic -- disabled for a test"
#     request.addfinalizer(fix_pheno)
#
#
# def test_cli_line_pheno_pass_1(example_timeseries, modify_config):
#     """ Run phenology
#     """
#     with modify_config(example_timeseries['config'],
#                        {'phenology': {'enable': True}}) as cfg:
#         runner = CliRunner()
#         result = runner.invoke(line.line, [cfg, '1', '5'],
#                                catch_exceptions=False)
#         assert result.exit_code == 0
#
#
# def test_cli_line_pheno_fail_1(example_timeseries, modify_config, break_pheno):
#     """ Run phenology, but mock it out so it fails
#     """
#     with modify_config(example_timeseries['config'],
#                        {'phenology': {'enable': True}}) as cfg:
#         runner = CliRunner()
#         result = runner.invoke(line.line, [cfg, '1', '5'],
#                                catch_exceptions=False)
#         assert result.exit_code == 1
#         assert 'Could not import' in result.output
