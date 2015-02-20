.. _guide_install:

============
Installation
============

Dependencies
------------

Yet Another TimeSeries Model (YATSM) is written in
`Python <https://www.python.org/>`_ (version 2.7)
and utilizes several C libraries, including the
`Geographic Data Abstraction Library <http://www.gdal.org/>`_ (GDAL).

These two dependencies are usually best installed through your package manager
on Linux, `Homebrew <http://brew.sh/>`_ on Mac OS, or through
`OSGEO4W <http://trac.osgeo.org/osgeo4w/>`_ on Windows.

For example, on Ubuntu 14.04:

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


virtualenv
----------

Python and GDAL are usually installed on most computing environments that
handle geographic data. If you wish to install YATSM on a system with these
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


Python Dependencies
--------------------

YATSM depends on many of the very common scientific Python modules. Because
`NumPy <http://www.numpy.org/>`_ is a build dependency for some of these
other modules, it must be installed first. With `pip`:

.. code-block:: bash

    $ pip install 'numpy>=1.9.1'


With NumPy installed, the remaining requirements may be installed through
`pip` from the `requirements.txt` file:

.. code-block:: bash

    $ pip install -r https://github.com/ceholden/yatsm/blob/master/requirements.txt


Quick Installation
------------------

YATSM may be installed from its
`Github repository <https://github.com/ceholden/yatsm>`_
using `pip`:

.. code-block:: bash

    $ pip install git+git://github.com/ceholden/yatsm.git


Developer Installation
----------------------

If you're interested in helping develop YATSM, or just forking it into your own
direction, you can download the repository using Git and build it locally:

.. code-block:: bash

    $ git clone https://github.com/ceholden/yatsm.git
    $ cd yatsm/
    $ python setup.py build_ext --inplace

After the Cython extensions are built using `setup.py`, YATSM will be usable
from this directory.

Documentation may be built using `Sphinx <http://sphinx-doc.org/>`_ from the
`docs` directory:

.. code-block:: bash

    $ cd docs/
    $ make html


Virtual Machine Image
---------------------

A lightweight Xubuntu 14.04 virtual machine image complete with all
dependencies and copies of YATSM and several other software useful for
remote sensing timeseries analysis, including
`TSTools <https://github.com/ceholden/TSTools/>`_ is available to download.

The virtual machine is formatted as a
`VirtualBox image <https://www.virtualbox.org/>`_
and I would recommend you to use
`VirtualBox <https://www.virtualbox.org/>`_ to run the virtual machine.
VirtualBox is a free and open source softare that can create and host virtual
machines and is comparable to commercial solutions such as VMWare or Parallels.

The virtual machine has been exported to a
`VirtualBox appliance <http://www.virtualbox.org/manual/ch01.html#ovf>`_
and uploaded to my university department's anonymous FTP server:

ftp://ftp-earth.bu.edu/ceholden/TSTools/

Please see the included README for further instructions.


Platform Support
----------------

YATSM is developed on Linux (CentOS 6 and Ubuntu 14.04) and has not been
tested on any other platforms, though I have seen it working on Mac OS. I am
welcome to any help fixing bugs or better supporting Windows, but I will not
try to support Windows myself.
