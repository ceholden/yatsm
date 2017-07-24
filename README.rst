Yet Another Timeseries Model (YATSM)
====================================

+----------+----------------+----------------------+
|          |  master_       | v0.6.x-maintenance_  |
+==========+================+======================+
| Build    | |build_master| | |build_v0.6.x|       |
+----------+----------------+----------------------+
| Coverage | |cov_master|   | |cov_v0.6.x|         |
+----------+----------------+----------------------+
| Docs     | |docs_master|  | |docs_v0.6.x|        |
+----------+----------------+----------------------+
| DOI      | |doi_master|   | |doi_v0.6.x|         |
+----------+----------------+----------------------+
|          | |Gitter|       |                      |
+--------------------------------------------------+


About
-----

Yet Another Timeseries Model (YATSM) is a Python package for utilizing a collection of timeseries algorithms and methods designed to monitor the land surface using remotely sensed imagery.

The `"Yet Another..." <http://en.wikipedia.org/wiki/Yet_another>`__ part of the package name is a reference to the algorithms implemented:

-  Continuous Change Detection and Classification (CCDC)

   -  Citation: Zhu and Woodcock, 2014; Zhu, Woodcock, Holden, and Yang 2015
   -  Note: Unvalidated, non-reference implementation

-  Long term mean phenology fitting using Landsat data

   -  Citation: Melaas, Friedl, and Zhu 2013
   -  Note: validated against Melaas *et al*'s code, but not a reference implementation

-  Commission detection via *p*-of-*F* (e.g., Chow test) test similar to what is used in LandTrendr (Kennedy, *et al*, 2010)
-  ...
-  More to come! Please reach out if you would like to help contribute

Note that the algorithms implemented within YATSM are not to be considered "reference" implementations unless otherwise noted.

The objective of making many methods of analyzing remote sensing timeseries available in one package is to leverage the strengths of multiple methods to overcome the weaknesses in any one approach. The opening of the Landsat archive in 2008 made timeseries analysis of Landsat data finally possible and kickstarted a "big bang" of methods that have evolved and proliferated since then. Over the years, it has become obvious that each individual algorithm is designed to monitor slightly different processes or leverages different aspects of the same datasets. Recent comparative analysis studies (Healey, Cohen, *et al*, forthcoming) strongly suggest that an ensemble of such algorithms is more accurate and informative than any one result alone. A suite of weak learners combined intelligently does indeed create a more powerful ensemble learner. By using a common set of vocabulary and making these algorithms available in one place, the YATSM package hopes to make such an ensemble possible.

Please consider citing as:

::

    Christopher E. Holden. (2017). Yet Another Time Series Model (YATSM). Zenodo. http://doi.org/10.5281/zenodo.251125

.. _master: https://github.com/ceholden/yatsm/tree/master
.. _v0.6.x-maintenance: https://github.com/ceholden/yatsm/tree/v0.6.x-maintenance
.. |build_master| image:: https://travis-ci.org/ceholden/yatsm.svg?branch=master
   :target: https://travis-ci.org/ceholden/yatsm
.. |cov_master| image:: https://coveralls.io/repos/ceholden/yatsm/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/ceholden/yatsm?branch=master&q=q
.. |docs_master| image:: https://readthedocs.org/projects/yatsm/badge/?version=master
   :target: https://yatsm.readthedocs.io/en/master/
   :alt: Documentation Status
.. |doi_master| image:: https://zenodo.org/badge/6804/ceholden/yatsm.svg
   :target: https://zenodo.org/badge/latestdoi/6804/ceholden/yatsm
.. |build_v0.6.x| image:: https://travis-ci.org/ceholden/yatsm.svg?branch=v0.6.x-maintenance
   :target: https://travis-ci.org/ceholden/yatsm
.. |cov_v0.6.x| image:: https://coveralls.io/repos/github/ceholden/yatsm/badge.svg?branch=v0.6.x-maintenance
   :target: https://coveralls.io/github/ceholden/yatsm?branch=v0.6.x-maintenance
.. |docs_v0.6.x| image:: https://readthedocs.org/projects/yatsm/badge/?version=v0.6.x-maintenance
   :target: https://yatsm.readthedocs.io/en/v0.6.x-maintenance/index.html
   :alt: Documentation Status
.. |doi_v0.6.x| image:: https://zenodo.org/badge/6804/ceholden/yatsm.svg
   :target: https://zenodo.org/badge/latestdoi/6804/ceholden/yatsm
.. |Gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/ceholden/yatsm?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=body_badge
