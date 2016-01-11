import fnmatch
from functools import partial
import os
import shutil
import tarfile
from tempfile import mkdtemp
try:
    from os import walk
except ImportError:
    from scandir import walk

if os.environ.get('TRAVIS'):
    # use agg backend on TRAVIS for testing so DISPLAY isn't required
    import matplotlib as mpl
    mpl.use('agg')

import numpy as np  # noqa
import pandas as pd  # noqa
import pytest  # noqa
import yaml  # noqa


here = os.path.dirname(__file__)
example_cachedir = os.path.join(here, 'data', 'cache')
example_cachefile = os.path.join(example_cachedir, 'yatsm_r0_n447_b8.npy.npz')
example_training = os.path.join(here, 'data', 'results',
                                'training_image_1995-06-01.gtif')
yaml_config = os.path.join(here, 'data', 'p035r032_config.yaml')

example_classify_config = 'RandomForest.yaml'
example_classify_pickle = 'train_rf.pkl'


# EXAMPLE DATASETS
@pytest.fixture(scope='session')
def example_timeseries(request):
    """ Extract example timeseries returning a dictionary of dataset attributes
    """
    path = mkdtemp('_yatsm')
    tgz = os.path.join(here, 'data', 'p035r032_testdata.tar.gz')
    with tarfile.open(tgz) as tgz:
        tgz.extractall(path)
    request.addfinalizer(partial(shutil.rmtree, path))

    # Find data
    subset_path = os.path.join(path, 'p035r032', 'images')
    stack_images, stack_image_IDs = [], []
    for root, dnames, fnames in walk(subset_path):
        for fname in fnmatch.filter(fnames, 'L*stack.gtif'):
            stack_images.append(os.path.join(root, fname))
            stack_image_IDs.append(os.path.basename(root))
    stack_images = np.asarray(stack_images)
    stack_image_IDs = np.asarray(stack_image_IDs)

    # Formulate "images.csv" input_file
    input_file = os.path.join(path, 'images.csv')
    dates = np.array([_d[9:16]for _d in stack_image_IDs])  # YYYYDOY
    sensors = np.array([_id[0:3] for _id in stack_image_IDs])  # Landsat IDs
    df = pd.DataFrame({
        'date': dates,
        'sensor': sensors,
        'filename': stack_images
    })
    # Sort by date
    pd_ver = pd.__version__.split('.')
    if pd_ver[0] == '0' and int(pd_ver[1]) < 17:
        df = df.sort(columns='date')
    else:
        df = df.sort_values(by='date')
    df.to_csv(input_file, index=False)

    # Copy configuration file
    dest_config = os.path.join(path, os.path.basename(yaml_config))
    config = yaml.load(open(yaml_config))
    config['dataset']['input_file'] = input_file
    config['dataset']['output'] = os.path.join(path, 'YATSM')
    config['dataset']['cache_line_dir'] = os.path.join(path, 'cache')
    config['classification']['training_image'] = example_training
    yaml.dump(config, open(dest_config, 'w'))

    return {
        'path': subset_path,
        'images': stack_images,
        'image_IDs': stack_image_IDs,
        'input_file': input_file,
        'images.csv': df,
        'config': dest_config,
    }


@pytest.fixture(scope='function')
def example_results(request, tmpdir):
    dst = os.path.join(tmpdir.mkdir('data').strpath, 'results')
    shutil.copytree(os.path.join(here, 'data', 'results'), dst)

    results = {
        'root': dst,
        'results_dir': os.path.join(dst, 'YATSM'),
        'results_dir_classified': os.path.join(dst, 'YATSM_classified'),
        'example_img': os.path.join(dst, 'example_image.gtif'),
        'classify_config': os.path.join(dst, example_classify_config),
        'example_classify_pickle': os.path.join(dst, example_classify_pickle)
    }
    return results


@pytest.fixture(scope='session')
def example_cache(request):
    return np.load(example_cachefile)


# EXAMPLE CACHE DATA
@pytest.fixture(scope='function')
def cachedir(request):
    return example_cachedir


@pytest.fixture(scope='function')
def cachefile(request):
    return example_cachefile


# MISC
@pytest.fixture(scope='function')
def mkdir_permissions(request):
    """ Fixture for creating dir with specific read/write permissions """
    def make_mkdir(read=False, write=False):
        if read and write:
            mode = 0755
        elif read and not write:
            mode = 0555
        elif not read and write:
            mode = 0333
        elif not read and not write:
            mode = 0000

        path = mkdtemp()
        os.chmod(path, mode)

        def fin():
            os.chmod(path, 0755)
            os.removedirs(path)
        request.addfinalizer(fin)

        return path

    return make_mkdir
