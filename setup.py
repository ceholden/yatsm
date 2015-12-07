import logging
import os
import shutil
import sys

from distutils.command.clean import clean as Clean
from setuptools import find_packages, setup
from setuptools.extension import Extension

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# Extra cleaning with MyClean
class MyClean(Clean):
    description = 'Remove files generated during build process'
    def run(self):
        Clean.run(self)
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

cmdclass = {'clean': MyClean}


# Get version
with open('yatsm/version.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

# Get README
with open('README.md') as f:
    readme = f.read()

# Build pickles
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, 'yatsm', 'regression', 'pickles'))
from yatsm.regression.pickles import serialize as serialize_pickles  # flake8: noqa
serialize_pickles.make_pickles()

# Installation requirements
install_requires = [
    'numpy',
    'scipy',
    'Cython',
    'statsmodels',
    'scikit-learn',
    'matplotlib',
    'click',
    'click_plugins',
    'palettable',
    'patsy'
]

# NumPy/Cython build setup
include_dirs = []
extra_compile_args = ['-O3']

try:
    import numpy as np
    include_dirs.append(np.get_include())
except ImportError:
    log.critical('NumPy and its headers are required for YATSM. '
                 'Please install and try again.')
    sys.exit(1)

try:
    from Cython.Build import cythonize
except ImportError:
    log.critical('Cython is required for YATSM. Please install and try again')
    sys.exit(1)

ext_opts = dict(
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args
)

ext_modules = cythonize([
    Extension('yatsm._cyprep', ['yatsm/_cyprep.pyx'], **ext_opts)
])

# Pre-packaged regression algorithms included in installation
package_data = {
    'yatsm': [
        os.path.join('regression', 'pickles', 'pickles.json'),
        os.path.join('regression', 'pickles', '*.pkl')
    ]
}

# Setup
packages = find_packages(exclude=['tests', 'yatsm.regression.pickles'])
packages.sort()

entry_points = '''
    [console_scripts]
    yatsm=yatsm.cli.main:cli

    [yatsm.yatsm_commands]
    cache=yatsm.cli.cache:cache
    pixel=yatsm.cli.pixel:pixel
    segment=yatsm.cli.segment:segment
    line=yatsm.cli.line:line
    train=yatsm.cli.train:train
    classify=yatsm.cli.classify:classify
    map=yatsm.cli.map:map
    changemap=yatsm.cli.changemap:changemap
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
    ext_modules=ext_modules,
    install_requires=install_requires,
    cmdclass=cmdclass
)

setup(**setup_dict)
