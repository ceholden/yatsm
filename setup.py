from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from yatsm.version import __version__

import numpy as np

with open('yatsm/version.py') as f:
    exec(f.read())

scripts = ['scripts/line_yatsm.py', 'scripts/run_yatsm.py',
           'scripts/gen_date_file.sh', 'scripts/train_yatsm.py',
           'scripts/yatsm_map.py', 'scripts/yatsm_changemap.py']

ext = [Extension('yatsm.cymasking', ['yatsm/cymasking.pyx'],
                 include_dirs=[np.get_include()])]

cmdclass = {'build_ext': build_ext}

setup(
    name='yatsm',
    version=__version__,
    author='Chris Holden',
    author_email='ceholden@gmail.com',
    packages=['yatsm'],
    scripts=scripts,
    url='https://github.com/ceholden/yatsm',
    license='LICENSE.txt',
    description='Land cover monitoring based on CCDC in Python',
    cmdclass=cmdclass,
    ext_modules=ext,
    install_requires=[
        'numpy',
        'scipy',
        'Cython',
        'statsmodels',
        'scikit-learn',
        'glmnet',
        'matplotlib',
        'docopt',
        'brewer2mpl'
    ]
)
