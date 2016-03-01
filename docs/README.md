YATSM  Documentation
--------------------

Documentation built using [Sphinx](http://sphinx-doc.org/) and hosted on [Github Pages](https://pages.github.com/).

## Building

### HTML

You can build the HTML files for the YATSM documentation using Sphinx. First, make sure the dependencies for YATSM's documentation are installed by `pip` installing the `docs/requirements.txt` file:

``` bash
pip install -r docs/requirements.txt
```

With Sphinx and other packages installed, use the `Makefile` to regenerate the HTML content:

``` bash
cd docs/
make html
```

You can now view the documentation you build using Sphinx by launching a web server and connecting to it using a web browser. For example, we can use the [`SimpleHTTPServer`](https://docs.python.org/2/library/simplehttpserver.html) or [`http.server`](https://docs.python.org/3/library/http.server.html) modules in Python 2 or 3, respectively, to host our documentation:

``` bash
python3 -m 'http.server'
# OR
python2 -m 'SimpleHTTPServer'
```

By default, your content will be hosted at `localhost:8000`.

### API

If you have written any code or changed the docstrings in any code, you will need to update the references to the code in the documentation. Regenerate API module information for Sphinx

``` bash
$ sphinx-apidoc -f -e -o docs/yatsm yatsm/
```

### Utilities

The documentation of command line utility and script usages can be automatically updated ``build_cli_docs.py``. This script writes the usages into ``cli/usage`` and can be included in documentation for scripts as follows:

``` rst
.. literalinclude:: usage/yatsm_line.rst
    :language: bash
```

Any additional text and documentation information can be included in the Restructured Text documents inside the `docs/scripts` folder.

### Guides

Guide information is written in Restructured Text and involves no auto-generated information.

## Publishing

Using [ghp-import](https://github.com/davisp/ghp-import) utility to push into `gh-pages` branch.

``` bash
[ ceholden@ceholden-llap: yatsm ]$ ghp-import -h
Usage: ghp-import [OPTIONS] DIRECTORY

Options:
  -n          Include a .nojekyll file in the branch.
  -m MESG     The commit message to use on the target branch.
  -p          Push the branch to origin/{branch} after committing.
  -r REMOTE   The name of the remote to push to. [origin]
  -b BRANCH   Name of the branch to write to. [gh-pages]
  -h, --help  show this help message and exit
```

Call:

```
ghp-import -n -m "$MSG" docs/_build/html
```
