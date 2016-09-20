.. _guide_models:

==============
Science Models
==============

The time series model run using the YATSM package command line interface (i.e., :ref:`yatsm_line` and :ref:`yatsm_pixel`) is specified in the configuration file using the ``YATSM['algorithm']`` key:

.. code-block:: yaml

    YATSM:
        algorithm: "AN_ALGORITHM"
        ...

    AN_ALGORITHM:
        init: ...
        fit: ...


The value specified for the ``algorithm`` key within the ``YATSM`` section must be the name of another section of the configuration file that can be used to identify, initialize, and run a time series algorithm included in the :mod:`yatsm.algorithms` module.


Models
------

.. toctree::
   :maxdepth: 2

   ccdcesque
