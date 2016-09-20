# -*- coding: utf-8 -*-
#
# YATSM documentation build configuration file, created by
# sphinx-quickstart on Tue Nov  4 18:26:04 2014.
import datetime as dt
import sys
import os
from os.path import abspath

project = u'YATSM'
copyright = u'2014 - {}, Chris Holden'.format(dt.datetime.utcnow().year)

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
d = os.path.dirname
sys.path.insert(0, d(d(d(abspath(__file__)))))

import yatsm
version = yatsm.__version__
release = yatsm.__version__

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
    'sphinx.ext.napoleon',
    'sphinxcontrib.bibtex',
    'sphinx.ext.githubpages',
    'sphinx_paramlinks'
]
autodoc_member_order = 'groupwise'

extlinks = {
    'issue': ('https://github.com/ceholden/yatsm/issues/%s', 'GH#'),
    'pr': ('https://github.com/ceholden/yatsm/pull/%s', 'GH#'),
    'commit': ('https://github.com/ceholden/yatsm/commits/master/%s', 'GH@')
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

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

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

_path = ['static']

html_show_sourcelink = False
