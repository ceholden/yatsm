""" Test ``yatsm changemap ...``
"""
from click.testing import CliRunner
import numpy as np

from yatsm.cli.main import cli

# flake8: ignore = E241

# TRUTH VALUES
first_change_YYYYDOY = np.array([
    [  -9999,   -9999, 2003288,   -9999,   -9999],
    [  -9999,   -9999,   -9999,   -9999,   -9999],
    [  -9999,   -9999, 2000152, 2001178, 2001146],
    [2001162,   -9999, 2001234, 2001234, 2001298],
    [2001234, 2001234, 2001194, 2001170, 2001266]], dtype=np.int32)
first_change_YYYYMMDD = np.array([
    [   -9999,    -9999, 20031015,    -9999,    -9999],
    [   -9999,    -9999,    -9999,    -9999,    -9999],
    [   -9999,    -9999, 20000531, 20010627, 20010526],
    [20010611,    -9999, 20010822, 20010822, 20011025],
    [20010822, 20010822, 20010713, 20010619, 20010923]], dtype=np.int32)

last_change_YYYYDOY = np.array([
    [  -9999,   -9999, 2003288,   -9999,   -9999],
    [  -9999,   -9999,   -9999,   -9999,   -9999],
    [  -9999,   -9999, 2000152, 2003192, 2010275],
    [2002229,   -9999, 2011182, 2001234, 2001298],
    [2001234, 2001234, 2010275, 2010275, 2010275]], dtype=np.int32)

num_change = np.array([
    [-9999, -9999,     1, -9999, -9999],
    [-9999, -9999, -9999, -9999, -9999],
    [-9999, -9999,     1,     2,     2],
    [    2, -9999,     2,     1,     1],
    [    1,     1,     2,     3,     3]], dtype=np.int32)


def test_changemap_first_pass_1(example_results, tmpdir, read_image):
    """ Make changemap of first change
    """
    image = tmpdir.join('first.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'changemap',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         'first', '2000-01-01', '2020-01-01', image
         ]
    )
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...], first_change_YYYYDOY)


def test_changemap_first_pass_2(example_results, tmpdir, read_image):
    """ Make changemap of first change, output as YYYYMMDD
    """
    image = tmpdir.join('first.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'changemap',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         '--out_date', '%Y%m%d',
         'first', '2000-01-01', '2020-01-01', image
         ]
    )
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...], first_change_YYYYMMDD)


def test_changemap_first_pass_3(example_results, tmpdir, read_image):
    """ Make changemap of first change, but with all NDV
    """
    image = tmpdir.join('first.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'changemap',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         '--out_date', '%Y%m%d',
         '--ndv', '0',
         'first', '1914-07-28', '1918-11-11', image
         ]
    )
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...],
                            np.zeros((5, 5), dtype=np.int16))


def test_changemap_last_pass_1(example_results, tmpdir, read_image):
    """ Make changemap of last change
    """
    image = tmpdir.join('last.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'changemap',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         'last', '2000-01-01', '2020-01-01', image
         ]
    )
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...], last_change_YYYYDOY)


def test_changemap_num_pass_1(example_results, tmpdir, read_image):
    """ Make changemap of number of changes
    """
    image = tmpdir.join('num.gtif').strpath
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ['-v', 'changemap',
         '--root', example_results['root'],
         '--result', example_results['results_dir'],
         '--image', example_results['example_img'],
         'num', '2000-01-01', '2020-01-01', image
         ]
    )
    assert result.exit_code == 0
    np.testing.assert_equal(read_image(image)[0, ...], num_change)
