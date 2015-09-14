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
2. ``YATSM`` describes model parameters common to all analysis and declares what change detection algorithm should be run
3. ``classification`` describes classification training data inputs
4. ``phenology`` describes phenology fitting parameters

The following tables describes the meanings of the parameter and values used
in the configuration file used in YATSM. Any parameters left blank will be
interpreted as ``None`` (e.g., ``cache_line_dir =``).

Dataset Parameters
------------------

.. note::

    This section is out of date for `v0.5.0` and requires re-writing

**Note**: you can use ``scripts/gen_date_file.sh`` to generate the CSV
file for ``input_file``.

Model Parameters
----------------

.. note::

    This section is out of date for `v0.5.0` and requires re-writing

Phenology
---------

The option for long term mean phenology calculation is an optional addition to `YATSM`. As such, visit :ref:`the phenology guide page <guide_phenology>` for configuration options.

Classification
--------------

The scripts included in YATSM which perform classification utilize a
configuration INI file that specify which algorithm will be used and the
parameters for said algorithm. The configuration details specified along
with the dataset and YATSM algorithm options deal with the training
data, not the algorithm details. These training data configuration
options include:

+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| Parameter                  | Data Type   | Explanation                                                                                                                             |
+============================+=============+=========================================================================================================================================+
| ``training_data``          | ``str``     | Training data raster image containing labeled pixels                                                                                    |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``mask_values``            | ``list``    | Values within the training data image to mask or ignore                                                                                 |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``training_start``         | ``str``     | Earliest date that training data are applicable. Training data labels will be paired with models that begin at least before this date   |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``training_end``           | ``str``     | Latest date that training data are applicable. Training data labels will be paired with models that end at least after this date        |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``training_date_format``   | ``str``     | Format specification that maps ``training_start`` and ``training_end`` to a Python datetime object (e.g., ``%Y-%m-%d``)                 |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+
| ``cache_xy``               | ``str``     | Filename used for caching paired X features and y training labels                                                                       |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+

Example
-------

An example template of the parameter file is located within
``examples/p013r030/p013r030.yaml``:

.. literalinclude:: ../../examples/p013r030/p013r030.yaml
   :language: yaml
