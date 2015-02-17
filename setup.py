from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

with open('yatsm/version.py') as f:
    for line in f:
        if line.find('__version__') >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"')
            version = version.strip("'")
            continue

with open('README.md') as f:
    readme = f.read()

scripts = ['scripts/line_yatsm.py',
           'scripts/run_yatsm.py',
           'scripts/gen_date_file.sh',
           'scripts/train_yatsm.py',
           'scripts/yatsm_map.py',
           'scripts/yatsm_changemap.py']

ext = [Extension('yatsm.cymasking',
                 ['yatsm/cymasking.pyx'],
                 include_dirs=[np.get_include()])]

cmdclass = {'build_ext': build_ext}

setup(
    name='yatsm',
    version=version,
    author='Chris Holden',
    author_email='ceholden@gmail.com',
    packages=['yatsm', 'yatsm.classifiers', 'yatsm.regression'],
    scripts=scripts,
    url='https://github.com/ceholden/yatsm',
    license='LICENSE.txt',
    description='Land cover monitoring based on CCDC in Python',
    long_description=readme,
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
