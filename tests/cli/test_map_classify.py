""" Test ``yatsm map classify ...``
"""
from click.testing import CliRunner
import numpy as np
import pytest

from yatsm.cli.main import cli


# CLASSIFICATION
classmap = np.array([[0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1],
                     [2, 2, 2, 2, 2],
                     [3, 3, 3, 3, 3],
                     [4, 4, 4, 4, 4]], dtype=np.uint8)
classmap_qa = np.array([[0, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3],
                        [3, 3, 3, 3, 3]], dtype=np.uint8)
classmap_proba = np.array([[0, 8000, 8000, 8000, 8000],
                           [8000, 8000, 8000, 8000, 8000],
                           [8000, 8000, 8000, 8000, 8000],
                           [8000, 8000, 8000, 8000, 8000],
                           [8000, 8000, 8000, 8000, 8000]], dtype=np.uint16)


def test_map_class_pass_1(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs
    """
    image = tmpdir.join('classmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir_classified'],
         '--image', example_results['example_img'],
         'class', '2005-06-01', image
         ])
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...], classmap)


def test_map_class_pass_2(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs, --before, --after switches
    """
    image = tmpdir.join('classmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir_classified'],
         '--image', example_results['example_img'],
         '--after', '--before',
         'class', '2005-06-01', image
         ])
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...], classmap)


def test_map_class_pass_3(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs, --before, --after, --qa switches
    """
    image = tmpdir.join('classmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir_classified'],
         '--image', example_results['example_img'],
         '--after', '--before', '--qa',
         'class', '2005-06-01', image
         ])
    assert result.exit_code == 0
    img = read_image(image)
    np.testing.assert_equal(img[0, ...], classmap)
    np.testing.assert_equal(img[1, ...], classmap_qa)


def test_map_class_pass_4(example_results, tmpdir, read_image):
    """ Make a map with reasonable inputs, --before, --after, --qa,
    --predict-proba switches
    """
    image = tmpdir.join('classmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir_classified'],
         '--image', example_results['example_img'],
         '--after', '--before', '--qa', '--predict-proba',
         'class', '2005-06-01', image
         ])
    assert result.exit_code == 0
    img = read_image(image)
    np.testing.assert_equal(img[0, ...], classmap)
    np.testing.assert_equal(img[1, ...], classmap_proba)
    np.testing.assert_equal(img[2, ...], classmap_qa)



def test_map_class_pass_5(example_results, tmpdir, read_image):
    """ Make a map with unreasonable date inputs
    """
    image = tmpdir.join('classmap.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'map',
         '--root', example_results['root'],
         '--result', example_results['results_dir_classified'],
         '--image', example_results['example_img'],
         '--after', '--before', '--qa',
         'class', '2005-06-01', image
         ])
    assert result.exit_code == 0
    img = read_image(image)
    np.testing.assert_equal(img[0, ...], classmap)
    np.testing.assert_equal(img[1, ...], classmap_qa)
