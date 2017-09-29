.. _guide_developer_custom_ts_algo:

===================================================
Integration of user-provided time series algorithms
===================================================

The YATSM package provides some time series algorithms as part of the
project, but developers can also build and integrate their own algorithms
into the suite of time series algorithms that can be utilized by programs like
:ref:`yatsm line <yatsm_line>` or :ref:`yatsm pixel <yatsm_pixel>`.

YATSM uses the setuptools_ module concept of "entry points" to
enumerate time series algorithms. YATSM algorithms are registered into a
group based on the type of algorithm -- there are entry point groups for
algorithms that find change as separate from algorithsm which postprocess
existing change results. Both categories link an entry point name to a
class or function usable within the YATSM package. For example, from the
``setup.py`` installation setup script:

.. code-block:: python

    [yatsm.algorithms.change]
    CCDCesque=yatsm.algorithms.ccdc:CCDCesque

This entry point definition links the name of the algorithm, "CCDCesque", to
the module (:mod:`yatsm.algorithms.ccdc`) containing the relevant
time series algorithm class, :class:`yatsm.algorithms.ccdc.CCDCesque`. Users
select the "CCDCesque" algorithm by defining it their configuration files.

Using entry points, you can create your own algorithms, distribute them in
separate Python packages, and YATSM will be able to find them and enable them
to work within the YATSM package.

Behind the scenes
-----------------

YATSM uses the function ``iter_entry_points`` from the setuptools_ module to
find and load all entry points associated with the YATSM package.

.. todo::

    Create example ``yatsm_algorithms`` repository to act as a template for
    including additional algorithms via setuptools_.


References:
-----------

- https://pythonhosted.org/setuptools/setuptools.html
- http://stackoverflow.com/questions/774824/explain-python-entry-points
- http://stackoverflow.com/questions/13545289/how-do-i-add-a-setuptools-entry-point-as-an-example-in-my-main-project
- https://docs.python.org/3/library/pkgutil.html


.. _pkgutil: https://docs.python.org/3/library/pkgutil.html
.. _setuptools: https://pythonhosted.org/setuptools/setuptools.html
