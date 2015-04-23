.. _index:

Yet Another Time Series Model (YATSM)
=====================================

About
-----

The Yet Another TimeSeries Model (YATSM) algorithm is designed to monitor land
surface phenomena, including land cover and land use change, using timeseries
of remote sensing observations. The algorithm seeks to find distinct time
periods within the timeseries, or time segments, by monitoring for disturbances.
These time segments may be used to infer continuous periods of stable land
cover, with breaks separating the segments representing ephemeral disturbances
or permanent conversions in land cover or land use.

The "Yet Another..." part of the algorithm name is an acknowledgement of the
influence a previously published timeseries algorithm - the Continuous Change
Detection and Classification (CCDC) :cite:`Zhu2014152` algorithm. While YATSM
began as an extension from CCDC, it was never intended as a 1 to 1 port of
CCDC and will continue to diverge in its own direction.

This algorithm is also influenced by other remote sensing algorithms which,
like CCDC, are based in theory on tests for structural change from econometrics
literature
:cite:`Chow1960,Andrews1993,Chu1996,Zeileis2005`.
These other remote sensing algorithms include
Break detection For Additive Season and Trend (BFAST) :cite:`Verbesselt201298`
and LandTrendr :cite:`Kennedy20102897`.
By combining ideas from CCDC, BFAST, and LandTrendr, this "Yet Another..."
algorithm hopes to overcome weaknesses present in any single approach.


User Guide
----------

To get started with YATSM, please follow this user guide:

.. toctree::
   :maxdepth: 2

   guide/install
   guide/dataset
   guide/exploration
   guide/model_specification
   guide/configuration
   guide/batch_interface
   guide/map_static
   guide/map_changes
   guide/classification
   guide/phenology


Command Line Utilities
----------------------

Documentation for individual command line applications from YATSM:

.. toctree::
   :maxdepth: 2

   scripts/scripts

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
----------

.. bibliography:: static/index_refs.bib
   :style: unsrt
   :cited:

