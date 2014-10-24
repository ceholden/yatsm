from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from yatsm.version import __version__

import numpy as np

with open('yatsm/version.py') as f:
    exec(f.read())

scripts = ['yatsm/line_yatsm.py', 'yatsm/run_yatsm.py']

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
        'cython >= 0.20.1',
        'numpy >= 1.8.1',
        'pandas >= 0.13.1',
        'statsmodels >= 0.5.0',
        'glmnet = 1.1-5',
        'scikit-learn >= 0.15.1',
        'ggplot >= 0.5.8',
        'docopt >= 0.6.1'
    ]
)
