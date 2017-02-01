import logging
import os
import shutil
import sys

from distutils.command.clean import clean as _clean
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop
from setuptools import find_packages, setup
from setuptools.extension import Extension

PY2 = sys.version_info[0] == 2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


def _build_pickles():
    # Build pickles
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, 'yatsm', 'regression', 'pickles'))
    from yatsm.regression.pickles import serialize as serialize_pickles  # noqa
    serialize_pickles.make_pickles()


# Extra cleaning with MyClean
class my_clean(_clean):
    description = 'Remove files generated during build process'

    def run(self):
        _clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('yatsm'):
            for filename in filenames:
                if any(filename.endswith(suffix) for suffix in
                       ('.c', '.so', '.pyd', '.pyc')):
                    os.unlink(os.path.join(dirpath, filename))
                    continue
                if (any(filename.endswith(suffix) for suffix in
                        ('.pkl', '.json')) and
                        os.path.basename(dirpath) == 'pickles'):
                    os.unlink(os.path.join(dirpath, filename))
            for dirname in dirnames:
                if dirname == '__pycache__':
                    shutil.rmtree(os.path.join(dirpath, dirname))


# Create pickles when building
class my_install(_install):
    def run(self):
        self.execute(_build_pickles, [], msg='Building estimator pickles')
        _install.run(self)


class my_develop(_develop):
    def run(self):
        self.execute(_build_pickles, [], msg='Building estimator pickles')
        _develop.run(self)


cmdclass = {
    'clean': my_clean,  # python setup.py clean
    'install': my_install,  # call when pip install
    'develop': my_develop  # called when pip install -e
}


# Get version
with open('yatsm/version.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue


# Get README
with open('README.rst') as f:
    readme = f.read()


# Installation requirements
extras_require = {
    'core': [
        'future', 'six',
        'numpy', 'pandas',
        'scipy',
        'matplotlib',
        'scikit-learn',
        'statsmodels',  # TODO: reevaluate need
        'patsy',
        'fiona',
        'rasterio',
        'xarray',
        'tables',
        'click',
        'click_plugins',
        'cligj',
        'pyyaml',
        'jsonschema'
    ],
    'pipeline': ['dask', 'distributed', 'toposort', 'graphviz']
}
if PY2:
    extras_require['pipeline'].extend(['futures'])
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))


# Pre-packaged regression algorithms included in installation
package_data = {
    'yatsm': [
        os.path.join('regression', 'pickles', 'pickles.json'),
        os.path.join('regression', 'pickles', '*.pkl'),
        os.path.join('config', 'config_schema.yaml')
    ]
}

# Setup
packages = find_packages(exclude=['tests', 'yatsm.regression.pickles'])
packages.sort()

entry_points = '''
    [console_scripts]
    yatsm=yatsm.cli.main:cli

    [yatsm.cli]
    batch=yatsm.cli.batch:batch
    cache=yatsm.cli.cache:cache
    pixel=yatsm.cli.pixel:pixel
    train=yatsm.cli.train:train
    classify=yatsm.cli.classify:classify
    map=yatsm.cli.map:map
    changemap=yatsm.cli.changemap:changemap

    [yatsm.algorithms.change]
    CCDCesque=yatsm.algorithms.ccdc:CCDCesque

    [yatsm.pipeline.tasks.tasks]
    norm_diff = yatsm.pipeline.tasks.preprocess:norm_diff

    [yatsm.pipeline.tasks.segment]
    pixel_CCDCesque = yatsm.pipeline.tasks.change:pixel_CCDCesque
'''


desc = ('Algorithms for remote sensing land cover and condition monitoring '
        'in Python')

setup_dict = dict(
    name='yatsm',
    version=version,
    author='Chris Holden',
    author_email='ceholden@gmail.com',
    packages=packages,
    package_data=package_data,
    include_package_data=True,
    entry_points=entry_points,
    url='https://github.com/ceholden/yatsm',
    license='MIT',
    description=desc,
    zip_safe=False,
    long_description=readme,
    install_requires=extras_require['core'],
    extras_require=extras_require,
    cmdclass=cmdclass
)
setup(**setup_dict)
