""" Test ``yatsm map coef ...``
"""
from click.testing import CliRunner
import numpy as np

from yatsm.cli.main import cli


# Truth for diagonals
diag = np.eye(5).astype(bool)
# Coefficient types
intercepts = np.arange(0, 7)
slopes = np.arange(7, 14)
seasonality_1 = np.arange(14, 21)
seasonality_2 = np.arange(21, 28)
dummies = np.arange(28, 35)
rmse = np.arange(35, 42)
all_coef = [intercepts, slopes, seasonality_1, seasonality_2, dummies, rmse]
# SWIR coefficients for Lasso20
coef_int_b5 = np.array([-9999., -16441.076172, 16221.29199219,
                        117207.890625, 393939.25], dtype=np.float32)
coef_slope_b5 = np.array([-9.99900000e+03, 2.333318e-02, -2.05697268e-02,
                          -1.55864090e-01, -5.34402490e-01], dtype=np.float32)
coef_season1_b5 = np.array([-9999., -3.76317239, -0., -0., -0.],
                           dtype=np.float32)
coef_season2_b5 = np.array([-9999., 112.587677, 165.92492676, 228.65888977,
                            255.87475586], dtype=np.float32)
coef_dummy_b5 = np.array([-9999., 0., -0., 0., -0.], dtype=np.float32)
coef_rmse_b5 = np.array([-9999., 113.20720673, 132.36845398, 140.73822021,
                         142.13438416], dtype=np.float32)
truths_b5 = [coef_int_b5, coef_slope_b5, coef_season1_b5, coef_season2_b5,
             coef_dummy_b5, coef_rmse_b5]
coef_amp_b5 = np.array([-9999., 112.650551, 165.92492676, 228.65888977,
                        255.87475586], dtype=np.float32)
# TODO: SWIR coefficients for OLS
# TODO: SWIR coefficients for RLM


# INTENTIONAL PASSES
def test_map_coef_pass_1(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs
    """
    image = tmpdir.join('coefmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         'coef', '2005-06-01', image
         ])
    img = read_image(image)
    assert result.exit_code == 0
    assert img.shape == (42, 5, 5)
    np.testing.assert_allclose(img[intercepts[4], diag], coef_int_b5)
    np.testing.assert_allclose(img[slopes[4], diag], coef_slope_b5)
    np.testing.assert_allclose(img[seasonality_1[4], diag], coef_season1_b5)
    np.testing.assert_allclose(img[seasonality_2[4], diag], coef_season2_b5)
    np.testing.assert_allclose(img[dummies[4], diag], coef_dummy_b5)
    np.testing.assert_allclose(img[rmse[4], diag], coef_rmse_b5)


def test_map_coef_pass_2(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs, selecting just one band
    """
    image = tmpdir.join('coefmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         '--band', '5',
         'coef', '2005-06-01', image
         ])
    img = read_image(image)
    assert result.exit_code == 0
    assert img.shape == (6, 5, 5)
    np.testing.assert_allclose(img[0, diag], coef_int_b5)
    np.testing.assert_allclose(img[1, diag], coef_slope_b5)
    np.testing.assert_allclose(img[2, diag], coef_season1_b5)
    np.testing.assert_allclose(img[3, diag], coef_season2_b5)
    np.testing.assert_allclose(img[4, diag], coef_dummy_b5)
    np.testing.assert_allclose(img[5, diag], coef_rmse_b5)


def test_map_coef_pass_3(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs, selecting just one band and coef
    """
    image = tmpdir.join('coefmap.gtif').strpath
    runner = CliRunner()

    idx_season = 0
    coefs = ['intercept', 'slope', 'seasonality', 'seasonality', 'categorical',
             'rmse']
    for coef, idx_coef, truth in zip(coefs, all_coef, truths_b5):
        result = runner.invoke(
            cli,
            ['-v', 'map',
             '--root', example_results['root'],
             '--result', example_results['results_dir'],
             '--image', example_results['example_img'],
             '--band', '5', '--coef', coef,
             'coef', '2005-06-01', image
             ])
        img = read_image(image)
        assert result.exit_code == 0
        if coef == 'seasonality':  # seasonality has 2 bands
            assert img.shape == (2, 5, 5)
            band_index = idx_season
            idx_season += 1
        else:
            band_index = 0
            assert img.shape == (1, 5, 5)
        np.testing.assert_allclose(img[band_index, diag], truth)


def test_map_coef_pass_amplitude(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs, selecting just one band and
    mapping only seasonality as amplitude
    """
    image = tmpdir.join('coefmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         '--band', '5', '--coef', 'seasonality', '--amplitude',
         'coef', '2005-06-01', image
         ])
    img = read_image(image)
    assert result.exit_code == 0
    assert img.shape == (1, 5, 5)
    np.testing.assert_allclose(img[0, diag], coef_amp_b5)


# OLS REFIT RESULTS
def test_map_coef_pass_refit_OLS(example_results, tmpdir, read_image):
    """ Make a map with refit OLS results
    """
    image = tmpdir.join('ols_refitmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         '--refit_prefix', 'ols',
         '--band', '5', '--coef', 'intercept',
         'coef', '2005-06-01', image
        ]
    )
    img = read_image(image)
    assert result.exit_code == 0
    assert img.shape == (1, 5, 5)



# INTENTIONAL FAILURES
def test_map_coef_fail_1(example_results, tmpdir, read_image):
    """ Error because of non-existent --image (trigger click.BadParameter)
    """
    image = tmpdir.join('coefmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', tmpdir.join('not_an_image.gtif').strpath,
         'coef', '2005-06-01', image
         ])
    assert result.exit_code == 2
    assert 'Cannot find example image' in result.output


def test_map_coef_fail_2(example_results, tmpdir, read_image):
    """ Error because of non-raster --image (trigger click.ClickException)
    """
    image = tmpdir.join('coefmap.gtif').strpath
    example = tmpdir.join('not_an_image.gtif').strpath
    with open(example, 'w') as f:
        f.write('some data')

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example,
         'coef', '2005-06-01', image
         ])
    assert result.exit_code == 1
    assert 'Could not open example image' in result.output
