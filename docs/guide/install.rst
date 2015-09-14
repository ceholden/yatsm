.. _guide_install:

============
Installation
============

Dependencies
------------

Yet Another TimeSeries Model (YATSM) is written in
`Python <https://www.python.org/>`_ (version 2.7)
and utilizes several C libraries, including the
`Geographic Data Abstraction Library <http://www.gdal.org/>`_ (GDAL). Package
dependencies are most easily installed using `Conda`_, but may also be
installed using your package manager on Linux (see `Ubuntu 14.04`_ example),
`Homebrew <http://brew.sh/>`_ on Mac OS, or through
`OSGEO4W <http://trac.osgeo.org/osgeo4w/>`_ on Windows.

Examples
--------

Conda
~~~~~

Requirements for YATSM may be installed using
|conda|_, Python's cross-platform and platform agnostic binary package
manager from `ContinuumIO <http://continuum.io/>`_. |conda|_ makes
installation of Python packages, especially scientific packages, very easy
because it includes compiled library dependencies that remove the need for a
compiler or pre-installed libraries.

Installation instructions for |conda|_ are available on their docs site
`conda.pydata.org <http://conda.pydata.org/docs/get-started.html>`_

Since |conda|_ makes installation so easy, installation through
|conda|_ will install all non-developer dependencies. Install YATSM using
|conda|_ into an isolated environment by using the ``environment.yaml`` file
as follows:

.. code-block:: bash

    # Install
    conda env create -n yatsm -f environment.yaml
    # Activate
    source activate yatsm


Ubuntu 14.04
~~~~~~~~~~~~

On Ubuntu 14.04, for example, the GDAL build dependencies may be satisfied
by installing the following:

.. code-block:: bash

    $ sudo apt-get install python2.7-dev
    $ sudo apt-get install gdal-bin libgdal-dev
    $ sudo apt-get install python-gdal

This installation guide will also utilize the
`pip <http://pip.readthedocs.org/en/latest/installing.html>`_ utility for
installing Python modules. `pip` may be installed or upgraded following the
instructions `here <http://pip.readthedocs.org/en/latest/installing.html>`_:

.. code-block:: bash

    $ wget https://bootstrap.pypa.io/get-pip.py
    $ python get-pip.py

On Ubuntu 14.04, it may be installed via the package manager:

.. code-block:: bash

    $ sudo apt-get install python-pip

You will also need two font packages for `matplotlib` that are not installed
with Ubuntu by default:

.. code-block:: bash

    $ sudo apt-get install libfreetype6-dev libxft-dev

With the GDAL library and `pip` installed, follow the guide for how to install
YATSM below using `virtualenv and PIP`_.

virtualenv and PIP
~~~~~~~~~~~~~~~~~~

Python and GDAL are usually installed on most computing environments that
handle geographic data (if not, see an example of installing these dependencies
on `Ubuntu 14.04`_). If you wish to install YATSM on a system with these
two dependencies but you don't have installation privileges for the Python
dependencies, you could install YATSM into a
`virtualenv <http://virtualenv.readthedocs.org/en/latest/>`_.

`virtualenv` creates isolated Python environments, thus enabling a user without
root privileges to install modules. `virtualenv` has the added benefit of
enabling the installation of specific module versions for single piece of
software.

To set up a `virtualenv`, it must first be available. With root access,

.. code-block:: bash

    $ sudo pip install virtualenv


Once `virtualenv` is installed, users may create a `virtualenv` for YATSM:

.. code-block:: bash

    $ virtualenv yatsm_venv

To activate this isolated Python environment from Bash:

.. code-block:: bash

    $ source yatsm_venv/bin/activate


Your terminal prompt will change, denoting the switch to this newly created
isolated Python environment.

YATSM Installation
------------------

YATSM may be installed from its
`Github repository <https://github.com/ceholden/yatsm>`_ using `pip`:

.. code-block:: bash

    $ pip install git+git://github.com/ceholden/yatsm.git


Developer Installation
----------------------

If you're interested in helping develop YATSM, you can download the repository using Git and install it in an editable installation:

.. code-block:: bash

    $ git clone https://github.com/ceholden/yatsm.git
    $ cd yatsm/
    $ pip install -e .

Documentation may be built using `Sphinx <http://sphinx-doc.org/>`_ from the
`docs` directory:

.. code-block:: bash

    $ cd docs/
    $ make html

Platform Support
----------------

YATSM is developed on Linux (CentOS 6 and Ubuntu 14.04) and has not been
tested on any other platforms, though I have seen it working on Mac OS. I am
welcome to any help fixing bugs or better supporting Windows, but I will not
try to support Windows myself.


.. |conda| replace:: ``conda``
.. _conda: http://conda.pydata.org/docs/get-started.html
