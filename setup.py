import logging
import os
import sys

from setuptools import find_packages, setup

PY2 = sys.version_info[0] == 2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()


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
        'rasterio',
        'fiona',
        'shapely',
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
        os.path.join('config', 'config_schema.yaml')
    ]
}

# Setup
packages = find_packages(exclude=['tests'])
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
)
setup(**setup_dict)
