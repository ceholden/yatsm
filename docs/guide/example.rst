.. _example_p013r030:

================
Example Workflow
================

This section of the guide covers the process of running of the CCDC
implementation (:py:class:`yatsm.algorithms.ccdc.CCDCesque`), including
the change detection and classification steps of the process, on
an example dataset centered on Harvard Forest in Masssachusetts.

It assumes you have already installed YATSM.

Data
----

The data used in this example are part of the collection of time series
subsets described in the `landsat_stack`_ repository. 

.. raw:: html

   <script src="https://embed.github.com/view/geojson/ceholden/landsat_stack/master/p013r030_bbox.geojson"></script>

Download at least the `p013r030_images.tar.bz2` as described in the repository
instructions. Place all downloads in the directory you want to work in
(preferably a new directory), and then unzip the files. The directory structure
created should look similar to the following:

.. code-block:: bash

   $ tree
   ├── p013r030
   │   ├── images
   │   │   ├── example_img
   │   │   ├── example_img.aux.xml
   │   │   ├── example_img.hdr
   │   │   ├── LE70130301999211EDC00
   │   │   │   ├── LE70130301999211EDC00_MTL.txt
   │   │   │   ├── LE70130301999211EDC00_stack
   │   │   │   ├── LE70130301999211EDC00_stack.aux.xml
   │   │   │   └── LE70130301999211EDC00_stack.hdr
   ...
   │   │   ├── LT50130302011316GNC01
   │   │   │   ├── LT50130302011316GNC01_MTL.txt
   │   │   │   ├── LT50130302011316GNC01_stack
   │   │   │   ├── LT50130302011316GNC01_stack.aux.xml
   │   │   │   └── LT50130302011316GNC01_stack.hdr
   │   │   └── YATSM
   │   │       ├── yatsm_r0.npz
   │   │       ├── yatsm_r100.npz
   ...
   │   └── maps
   │       ├── p013r030_HF_change_first_1985-2016.gtif
   │       ├── p013r030_HF_change_first_1985-2016.gtif.aux.xml
   │       ├── p013r030_HF_change_last_1985-2016.gtif
   │       ├── p013r030_HF_change_last_1985-2016.gtif.aux.xml
   ...
   │       └── p013r030_maps.qgs
   ├── p013r030_images.tar.bz2
   ├── p013r030_maps.tar.gz
   └── p013r030_results.tar.gz


Setup
-----

To use any of the built-in programs that facilitate running time series
analysis, we need to describe the time series data using a CSV file and
we need to provide our analysis parameters using a configuration file
(specified in the YAML format). 

To begin, copy the example configuration file located within this repository
as ``examples/p013r030/p013r030.yaml``, or by downloading it from Github
(`p013r030.yaml`_).

Next, we will use the helper script located at ``scripts/gen_date_file.sh``
to generate a CSV file with the columns ``date`` and ``filename``
and rows for each image we want to include in our analysis. For example,

.. code-block:: bash

     date,filename
     1984162,images/LT50130301984162XXX08/LT50130301984162XXX08_stack
     1984274,images/LT50130301984274PAC08/LT50130301984274PAC08_stack
     1984290,images/LT50130301984290PAC16/LT50130301984290PAC16_stack

You can use the Bash helper script, located in the repository as
``scripts/gen_date_file.sh``, to create this CSV file. If this script
doesn't work for your data, or you can't run Bash scripts, generate
this information using some other means.

.. code-block:: bash

   $ scripts/gen_date_file.sh -v images images.csv
   Searching in images
   Searching for pattern: L*stack
   YYYYDOY starts at 9
   Output file is images.csv
   Found 423 images


Next, open the configuration file with a text editor and make some changes
under the ``dataset:`` section to point to this newly generated ``images.csv``
file (adjust this name if you named the CSV file something different). Change
the value of the ``input_file`` parameter. You will also want to change the
values for ``output`` and ``cache_line_dir``, which control the location
of output result files and "cached data" files, respectively.

.. code-block:: yaml
   :emphasize-lines: 3,7,23

   dataset:
       # Text file containing dates and images
       input_file: "images.csv"
       # Input date format
       date_format: "%Y%j"
       # Output location
       output: "images/YATSM"
       # Output file prefix (e.g., [prefix]_[line].npz)
       output_prefix: "yatsm_r"
       # Total number of bands
       n_bands: 8
       # Mask band (e.g., Fmask)
       mask_band: 8
       # List of integer values to mask within the mask band
       mask_values: [2, 3, 4, 255]
       # Valid range of band data
       # specify 1 range for all bands, or specify ranges for each band
       min_values: 0
       max_values: 10000
       # Use BIP image reader? If not, use GDAL to read in
       use_bip_reader: False
       # Directory location for caching dataset lines
       cache_line_dir: "cache"

Pixel Plotter
-------------

A good first test to make sure your dataset configuration was successful,
as well as a good place to start for experimenting with different runtime
parameters, is using the ``yatsm pixel`` command

`yatsm pixel <../cli/yatsm_pixel>`_


Batch Change Detection Processing
---------------------------------

The next step in the process is to run the change detection on all
pixels in your dataset.

:ref:`yatsm line <yatsm_line>`

Change Detection Visualization
------------------------------

:ref:`yatsm changemap <yatsm_changemap>`

Classification Training
-----------------------

Find training data, remembering to record the range of time for which
the training data is accurate or relevant for so we will know which
time series segment attributes (``X``) to match with the training data
labels (``y``).

:ref:`yatsm train <yatsm_train>`


Classification Prediction
-------------------------

Using the trained classifier, classify the time series segments identified
in the change detection step.

:ref:`yatsm classify <yatsm_classify>`

Land Cover Mapping
------------------

:ref:`yatsm map <yatsm_map>`




.. _`landsat_stack`: https://github.com/ceholden/landsat_stack
.. _`p013r030.yaml`: https://raw.githubusercontent.com/ceholden/yatsm/master/examples/p013r030/p013r030.yaml
