# -*- coding: utf-8 -*-
#
# YATSM documentation build configuration file, created by
# sphinx-quickstart on Tue Nov  4 18:26:04 2014.
import datetime as dt
import sys
import os

import sphinx

try:
    from unittest import mock
    from unittest.mock import MagicMock
except:
    import mock
    from mock import MagicMock
mock.FILTER_DIR = False


MOCK_MODULES = [
    'glmnet',
    'matplotlib', 'matplotlib.cm', 'matplotlib.pyplot', 'matplotlib.style',
    'numpy', 'numpy.lib', 'numpy.lib.recfunctions', 'numpy.ma',
    'numba',
    'osgeo',
    'pandas',
    'patsy',
    'rpy2', 'rpy2.robjects', 'rpy2.robjects.numpy2ri',
    'rpy2.robjects.packages',
    'scipy', 'scipy.ndimage', 'scipy.stats',
    'sklearn', 'sklearn.cross_validation', 'sklearn.ensemble',
    'sklearn.externals', 'sklearn.externals.joblib',
    'sklearn.linear_model', 'sklearn.utils',
    'statsmodels', 'statsmodels.api',
    'yatsm._cyprep'
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
d = os.path.dirname
sys.path.insert(0, d(d(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(d(d(os.path.abspath(__file__))), 'scripts'))

# -- General configuration ------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosummary',
    'sphinx.ext.autodoc',
    'sphinx.ext.extlinks',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinxcontrib.bibtex',
    'sphinx_paramlinks'
]
autodoc_member_order = 'groupwise'
extlinks = {
    'issue': ('https://github.com/ceholden/yatsm/issues/%s', 'issue ')
}
intersphinx_mapping = {
    'sklearn': ('http://scikit-learn.org/stable', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'setuptools': ('https://pythonhosted.org/setuptools/', None),
    'python': ('https://docs.python.org/dev', None)
}
todo_include_todos = True
graphviz_output_format = "svg"
graphviz_dot_args = ['-Gratio="compress"']

latex_elements = {
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{bm}
        \usepackage{color}
    '''
}

# Napoleon extension moving to sphinx.ext.napoleon as of sphinx 1.3
sphinx_version = sphinx.version_info
if sphinx_version[0] >= 1 and sphinx_version[1] >= 3:
    extensions.append('sphinx.ext.napoleon')
else:
    extensions.append('sphinxcontrib.napoleon')
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'YATSM'
copyright = u'2014 - {}, Chris Holden'.format(dt.datetime.utcnow().year)

import yatsm  # noqa
version = yatsm.__version__
release = yatsm.__version__
html_last_updated_fmt = '%c'

exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# -- Options for HTML output ----------------------------------------------

# on_rtd is whether we are on readthedocs.org, this line of code grabbed from
# docs.readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_context = dict(
    display_github=True,
    github_user="ceholden",
    github_repo="yatsm",
    github_version="master",
    conf_py_path="/docs/",
    source_suffix=".rst",
)

# html_theme_options = { }

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
_path = ['static']

# Add any extra paths that contain custom files (such as robots.txt or
# .htaccess) here, relative to this directory. These files are copied
# directly to the root of the documentation.
# html_extra_path = []

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''
