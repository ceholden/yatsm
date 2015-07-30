import logging
import sys

from setuptools import setup
from setuptools.extension import Extension

logging.basicConfig()
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
with open('README.md') as f:
    readme = f.read()

# Installation requirements
install_requires = [
    'numpy',
    'scipy',
    'Cython',
    'statsmodels',
    'scikit-learn',
    'glmnet',
    'matplotlib',
    'docopt',
    'brewer2mpl',
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

# Setup
packages = ['yatsm', 'yatsm.cli',
            'yatsm.classifiers', 'yatsm.regression', 'yatsm.segment']

scripts = ['scripts/run_yatsm.py',
           'scripts/gen_date_file.sh',
           'scripts/train_yatsm.py',
           'scripts/yatsm_map.py',
           'scripts/yatsm_changemap.py']

entry_points = '''
    [console_scripts]
    yatsm=yatsm.cli.main:cli

    [yatsm.yatsm_commands]
    segment=yatsm.cli.segment:segment
    line=yatsm.cli.line:line
'''

setup_dict = dict(
    name='yatsm',
    version=version,
    author='Chris Holden',
    author_email='ceholden@gmail.com',
    packages=packages,
    scripts=scripts,
    entry_points=entry_points,
    url='https://github.com/ceholden/yatsm',
    license='MIT',
    description='Land cover monitoring based on CCDC in Python',
    long_description=readme,
    ext_modules=ext_modules,
    install_requires=install_requires
)

setup(**setup_dict)
