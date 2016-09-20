Change Log
==========

All notable changes will appear in this log that begins with the release of ``v0.5.0``. Changes are categorized into "Added", "Changed", "Fixed", and "Removed". To see a comparison between releases on Github, click or follow the release version number URL.

For information on the style of this change log, see `keepachangelog.com <http://keepachangelog.com/>`__.

`v0.6.1 <https://github.com/ceholden/yatsm/compare/v0.6.0...v0.6.1>`__ - 2016-05-12
-----------------------------------------------------------------------------------

Version ``v0.6.x`` will be backward patched for any bug fixes (for an undetermined amount of time) as version ``v0.7.0`` will introduce backwards incompatible changes in order to enable incorporation of data from multiple sensors and to better link time series models together in a cohesive pipeline.

Fixed
~~~~~

- ``CCDCesque``: Fixed case in which bands not used as "test indices" would not have time series models estimated (i.e., no `coef` or `rmse`) if the time series ends immediately after training `#88 <https://github.com/ceholden/yatsm/issues/88>`_
- ``CCDCesque``: Fixed for case when a model refit would try to take place despite n < p (`commit <https://github.com/ceholden/yatsm/commit/5c27bad3f394e35166ae94e3663692ecd7bcfe43>`__)
- ``RLM``: Fixed divide by zero error when ``n == p`` (number of observations equals number of parameters estimated)

`v0.6.0 <https://github.com/ceholden/yatsm/compare/v0.5.5...v0.6.0>`__ - 2016-04-22
-----------------------------------------------------------------------------------

`Milestone
v0.6.0 <https://github.com/ceholden/yatsm/milestones/v0.6.0>`__

Changed
~~~~~~~

-  ``CCDCesque``: Optimize algorithm implementation. Performance estimates show 2x speed gain `#70 <https://github.com/ceholden/yatsm/issues/70>`__
-  CLI: Improve ``yatsm pixel`` by enabling the plotting of multiple refit model estimates on the same graph (`commit <https://github.com/ceholden/yatsm/commit/0e6e1e5265e2786588b2cddf061693880cbe2e3c>`__)
-  CLI: Improve ``yatsm pixel`` ``--embed`` option (`commit <https://github.com/ceholden/yatsm/commit/b1cf47ff3feeeb93b9f671bccc4379a9da1ad808>`__)
-  CLI: Add ``--verbose-yatsm`` to main ``yatsm`` command so it works with all programs running a YATSM algorithm (`commit <https://github.com/ceholden/yatsm/commit/772badc980c56d2d5c4185a40bf856bc6875be91>`__)
-  Use ``setuptools`` entry points to point YATSM to available time series algorithms (`commit <https://github.com/ceholden/yatsm/commit/a30424e044391062150851e566100bec4df66623>`__)

Added
~~~~~

-  Expose ``stay_regularized`` for segment refitting steps `#74 <https://github.com/ceholden/yatsm/issues/74>`__
-  Add capability to specify ``fit`` section for statistical estimators that are passed to the ``fit`` method of the estimator `#61 <https://github.com/ceholden/yatsm/issues/61>`__
-  ``CCDCesque``: allow specification of ``min_rmse`` per band using an array or just one value for all bands `#75 <https://github.com/ceholden/yatsm/issues/75>`__
-  Add submodule ``yatsm.regression.diagnostics`` for regression diagostics, including RMSE (`commit <https://github.com/ceholden/yatsm/commit/df582d235a6e6c8e114053015a7b7392bee8f570>`__)
-  Add new module ``yatsm.accel`` with decorator (``try_jit``) that applies ``numba.jit`` to functions only if ``numba`` is available `#70 <https://github.com/ceholden/yatsm/issues/70>`__
-  Apply ``yatsm.accel.try_jit`` to calculation of ``yatsm.regression.diagnostics.rmse``,
   ``yatsm.regression.robust_fit.RLM``, and others `#70 <https://github.com/ceholden/yatsm/issues/70>`__
-  Benchmark algorithm performance across project history using `Airspeed Velocity <https://github.com/spacetelescope/asv/>`__ `#71 <https://github.com/ceholden/yatsm/issues/71>`__
-  Improve ``clean`` target in package's ``setup.py`` so it deletes built estimator pickles and ``.c``/``.so`` built with Cython (`commit <https://github.com/ceholden/yatsm/commit/bb868922a2f6f2f68c9f71153c4307e8727468cb>`__)
-  Increase test coverage from ~20% to ~80%
-  Added documentation to `Read the Docs <readthedocs.org>`_

Fixed
~~~~~

-  ``CCDCesque``: Fix bug in calculation of ``end`` attribute for last timeseries record
   `#72 <https://github.com/ceholden/yatsm/issues/72>`__
-  ``CCDCesque``: Fix bug in parsing of ``test_indices`` if user doesn't supply any `#73 <https://github.com/ceholden/yatsm/issues/73>`__
-  "Packaged" estimator pickles are built on installation of YATSM so they will work with user versions of libraries (`commit <https://github.com/ceholden/yatsm/commit/d9b4b80c1c70137525abfde7fc7933e34bcf6820>`__)
-  Fix ``DeprecationWarnings`` with ``scikit-learn>=0.17.0`` (`commit <https://github.com/ceholden/yatsm/commit/29ddd4c0da29904b49fca7e452ee23ca1f938261>`__)
- ``yatsm.regression.robust_fit.RLM``: Fix a bug caused by dividing by zero. This bug only occurs when the number of observations in a time series segment is approximately equal to the number of parameters (``n ~= k``) `#86 <https://github.com/ceholden/yatsm/issues/86>`_
- Fix NumPy deprecation warnings and improve ``yatsm changemap num`` performance `#83 <https://github.com/ceholden/yatsm/issues/83>`__

`v0.5.5 <https://github.com/ceholden/yatsm/compare/v0.5.4...v0.5.5>`__ - 2015-11-24
-----------------------------------------------------------------------------------

`Milestone v0.5.5 <https://github.com/ceholden/yatsm/milestones/v0.5.5>`__

Added
~~~~~

-  Abort if config file 'n\_bands' looks incorrect (`commit <https://github.com/ceholden/yatsm/commit/01a6adec1fcd567c194e28b98fa488c13cdbdd45>`__)

Changed
~~~~~~~

-  Reorganize long term mean phenology code into generic phenology related submodule.
-  Reorganize changemap and map logic to separate module `#60 <https://github.com/ceholden/yatsm/issues/60>`__

Fixed
~~~~~

-  Fix bug with spline EVI prediction in LTM phenology module when data include last day in leap year (366) `#56 <https://github.com/ceholden/yatsm/issues/56>`__
-  Fix bug with phenology half-max calculation that created erroneous transition dates `#58 <https://github.com/ceholden/yatsm/issues/58>`__
-  Fix bug with phenology calculation for 100% masked data pixels `#54 <https://github.com/ceholden/yatsm/issues/54>`__
-  Fix ``yatsm pixel`` to correctly plot designs that include categorical variables (`commit <https://github.com/ceholden/yatsm/commit/966edd8b4a95e3c19d677eb71e2b76a155911d88>`__)
-  Fix passing of a list of dataset min/max values within config files instead of 1 number `#59 <https://github.com/ceholden/yatsm/issues/59>`__
-  Add missing ``phenology`` module to ``setup.py`` (`commit <https://github.com/ceholden/yatsm/commit/9d49d737316b34d2465b18db55647d7104d17758>`__)
`v0.5.4 <https://github.com/ceholden/yatsm/compare/v0.5.3...v0.5.4>`__ - 2015-10-28
-----------------------------------------------------------------------------------

`Milestone v0.5.4 <https://github.com/ceholden/yatsm/milestones/v0.5.4>`__

Fixed
~~~~~

-  Fix multiple bugs encountered when running phenology estimates `#49 <https://github.com/ceholden/yatsm/issues/49>`__

Changed
~~~~~~~

-  Metadata from ``yatsm line`` runs are now stored in ``metadata`` sub-file of NumPy compressed saved files `#53 <https://github.com/ceholden/yatsm/issues/53>`__
-  Algorithm configurations must now declare subsections that match estimator methods (e.g., ``init`` and ``fit``) `#52 <https://github.com/ceholden/yatsm/issues/52>`__
-  Refactored ``yatsm.phenology`` to make ``LongTermMeanPhenology`` estimator follow ``scikit-learn`` API `#50 <https://github.com/ceholden/yatsm/issues/50>`__

Added
~~~~~

-  Add ``--num_threads`` option to ``yatsm`` CLI. This argument sets various environment variables (e.g., ``OPENBLAS_NUM_THREADS`` or ``MKL_NUM_THREADS``) before beginning computation to set or limit multithreaded linear algebra calculations within NumPy `#51 <https://github.com/ceholden/yatsm/issues/51>`__
-  Add this changelog!

`v0.5.3 <https://github.com/ceholden/yatsm/compare/v0.5.2...v0.5.3>`__ - 2015-10-20
-----------------------------------------------------------------------------------

`Milestone v0.5.3 <https://github.com/ceholden/yatsm/milestones/v0.5.3>`__

Changed
~~~~~~~

-  Update configuration file parsing for classifiers to use ``yaml``. Users need to update their classifier configuration files to use this new format.

Fixed
~~~~~

-  Fixed bug when running on real datasets with 100% missing data in timeseries (e.g., in scene corners) `#47 <https://github.com/ceholden/yatsm/issues/47>`__ `#48 <https://github.com/ceholden/yatsm/issues/48>`__
-  Fix ``yatsm train`` and ``yatsm classify`` for ``v0.5.0+`` releases

Removed
~~~~~~~

-  Deleted intermediate "helper" classes that were used to type-check ``ini`` configuration files

`v0.5.2 <https://github.com/ceholden/yatsm/compare/v0.5.1...v0.5.2>`__ - 2015-10-09
-----------------------------------------------------------------------------------

`Milestone v0.5.2 <https://github.com/ceholden/yatsm/milestones/v0.5.2>`__

Fixed
~~~~~

-  Catch ``TSLengthException`` so ``yatsm line`` can continue running `#43 <https://github.com/ceholden/yatsm/issues/43>`__
-  Allow refit estimators to be from pre-packaged, distributed pickles `#44 <https://github.com/ceholden/yatsm/issues/44>`__
-  Remove references to old variable names in ``yatsm.algorithms.postprocess`` `#45 <https://github.com/ceholden/yatsm/issues/45>`__

`v0.5.1 <https://github.com/ceholden/yatsm/compare/v0.5.0...v0.5.1>`__ - 2015-10-06
-----------------------------------------------------------------------------------

`Milestone v0.5.1 <https://github.com/ceholden/yatsm/milestones/v0.5.1>`__

Added
~~~~~

-  Use environment variables in configuration files `#42 <https://github.com/ceholden/yatsm/issues/42>`__
-  Pre-package a set of pickled estimators using ``package_data`` from ``setuptools`` `#41 <https://github.com/ceholden/yatsm/issues/41>`__

v0.5.0 - 2015-09-14
-------------------

`Milestone v0.5.0 <https://github.com/ceholden/yatsm/milestones/v0.5.0>`__

Very backwards incompatible release required to redefine project objectives and use better technology (``click`` & ``YAML``) for command line interface.

Changed
~~~~~~~

-  Command line interface uses `click <click.pocoo.org>`__ `#28 <https://github.com/ceholden/yatsm/issues/28>`__
-  Redefine ``YATSM`` as baseclass and rename CCDC implementation to ``CCDCesque`` `#29 <https://github.com/ceholden/yatsm/issues/28>`__
-  Specify prediction method using serialized "pickle" instances of ``scikit-learn`` compatible estimators `#26 <https://github.com/ceholden/yatsm/issues/26>`__
-  Configuration file now uses ``YAML`` format for better organization and more sustainable parsing `#30 <https://github.com/ceholden/yatsm/issues/30>`__
-  Refactor ``robust`` fit into more generalized ``refit`` step. User can generate additional ``[prefix]_coef`` and ``[prefix]_rmse`` results using specified estimators `#33 <https://github.com/ceholden/yatsm/issues/33>`__
-  Tests now use ``py.test`` fixtures for better code reuse
-  Reorganize ``requirements.txt`` organization and documentation

Added
~~~~~

-  Add ``environment.yaml`` for creating environments within the `Anaconda <https://www.continuum.io/downloads>`__ distribution using ``conda``
