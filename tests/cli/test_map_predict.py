""" Test ``yatsm map predict ...``
"""
from click.testing import CliRunner
import numpy as np

from yatsm.cli.main import cli


# Truth for diagonals
diag = np.eye(5).astype(bool)
# SWIR answers
BAND_SWIR = 4
pred_swir = np.array([-9999, 723, 1279, 3261, 2885], dtype=np.int16)


def test_map_predict_pass_1(example_results, tmpdir, read_image):
    """ Make a map of predictions
    """
    image = tmpdir.join('predict.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli, [
            '-v', 'map',
            '--root', example_results['root'],
            '--result', example_results['results_dir'],
            '--image', example_results['example_img'],
            'predict', '2005-06-01', image
        ]
    )
    img = read_image(image)
    assert result.exit_code == 0
    assert img.shape == (7, 5, 5)
    np.testing.assert_equal(img[BAND_SWIR, diag], pred_swir)
