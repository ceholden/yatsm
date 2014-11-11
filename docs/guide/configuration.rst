.. _model_config:

Model Configuration
===================

Configuration File
------------------

The batch running script uses an `INI
file <https://wiki.python.org/moin/ConfigParserExamples>`_ to
parameterize the run. The INI file uses three sections:

1. ``[dataset]`` describes dataset attributes
2. ``[YATSM]`` describes model parameters
3. ``[classification]`` describes classification training data inputs

The following tables describes the meanings of the parameter and values used
in the configuration file used in YATSM. Any parameters left blank will be
interpreted as ``None`` (e.g., ``cache_line_dir =``).

Dataset Parameters
------------------

The following dataset information is required:

+----------------------+----------------+----------------------------------------------------------------------------------------------+
| Parameter            | Data Type      | Explanation                                                                                  |
+======================+================+==============================================================================================+
| ``input_file``       | ``filename``   | The filename of a CSV file recording the date and filenames of all images in the dataset     |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``date_format``      | ``str``        | The format of the dates specified in the ``input_file`` (e.g., ``%Y%j``)                     |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``output``           | ``str``        | Output folder location for results                                                           |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``output_prefix``    | ``str``        | Output file prefix (e.g., ``[prefix]_[line].npz``)                                           |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``n_bands``          | ``int``        | The number of bands in the images                                                            |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``mask_band``        | ``int``        | Band index in each image of the mask band                                                    |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``green_band``       | ``int``        | Band index in each image of the green band (~520-600 nm)                                     |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``swir1_band``       | ``int``        | Band index in each image of the shortwave-infrared band (~1550-1750 nm)                      |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``use_bip_reader``   | ``bool``       | Use ``fopen`` style read in for band interleave by pixel (BIP) files, instead of GDAL's IO   |
+----------------------+----------------+----------------------------------------------------------------------------------------------+
| ``cache_line_dir``   | ``str``        | Directory location for caching dataset lines                                                 |
+----------------------+----------------+----------------------------------------------------------------------------------------------+

**Note**: you can use ``scripts/gen_date_file.sh`` to generate the CSV
file for ``input_file``.

Model Parameters
----------------

The change detection is parameterized by:

+----------------------+-------------+---------------------------------------------------------------------------------+
| Parameter            | Data Type   | Explanation                                                                     |
+======================+=============+=================================================================================+
| ``consecutive``      | ``int``     | Consecutive observations to trigger change                                      |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``threshold``        | ``float``   | Test statistic critical value to trigger change                                 |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``min_obs``          | ``int``     | Minimum observations per time segment in model                                  |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``min_rmse``         | ``float``   | Minimum RMSE in test statistic for each model                                   |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``freq``             | ``list``    | Frequency of sine/cosine seasonal periods                                       |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``test_indices``     | ``list``    | Indices of Y to use in change detection                                         |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``screening``        | ``str``     | Method for screening timeseries for noise. Options are ``RLM`` and ``LOWESS``   |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``screening_crit``   | ``float``   | Critical value for detecting noise in multitemporal cloud/shadow screening      |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``lassocv``          | ``bool``    | Use ``sklearn.linear_model.LassoLarsCV`` instead of ``glmnet``                  |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``reverse``          | ``bool``    | Run model backward in time, rather than forward                                 |
+----------------------+-------------+---------------------------------------------------------------------------------+
| ``robust``           | ``bool``    | Return coefficients and RMSE from a robust linear model for each time segment   |
+----------------------+-------------+---------------------------------------------------------------------------------+

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
| ``cache_Xy``               | ``str``     | Filename used for caching paired X features and y training labels                                                                       |
+----------------------------+-------------+-----------------------------------------------------------------------------------------------------------------------------------------+

Example
-------

An example template of the parameter file is located within
``examples/example.ini``:

.. include:: ../../examples/example.ini
   :literal:
