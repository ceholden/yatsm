.. _guide_model_config:

===================
Model Configuration
===================

The issue tracker on Github is being used to track additions to this
documentation section. Please see
`ticket 37 <https://github.com/ceholden/yatsm/issues/37>`_.


Configuration File
------------------

The batch running script uses an `YAML
file <https://en.wikipedia.org/wiki/YAML>`_ to
parameterize the run. The YAML file uses several sections:

1. ``dataset`` describes dataset attributes common to all analysis
2. ``YATSM`` describes model parameters common to all analysis and declares what
   change detection algorithm should be run
3. ``classification`` describes classification training data inputs
4. ``phenology`` describes phenology fitting parameters

The following tables describes the meanings of the parameter and values used
in the configuration file used in YATSM. Any parameters left blank will be
interpreted as ``None`` (e.g., ``cache_line_dir =``).

Example
-------

An example template of the parameter file is located within
``examples/p013r030/p013r030.yaml``:

.. literalinclude:: ../../examples/p013r030/p013r030.yaml
   :language: yaml
