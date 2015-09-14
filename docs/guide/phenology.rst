.. _guide_phenology:

========================
Long Term Mean Phenology
========================

This part of the guide is a work in progress and more information will be added.

The purpose of this module within YATSM is to provide long term mean phenological
characteristics of landscapes using an algorithm developed by Melaas *et al*
(2013) for each stable land cover segment identified by the CCDC portion of YATSM.

See:

    Melaas, EK, MA Friedl, and Z Zhu. 2013. Detecting interannual variation in
    deciduous broadleaf forest phenology using Landsat TM/ETM+ data. Remote
    Sensing of Environment 132: 176-185. http://dx.doi.org/10.1016/j.rse.2013.01.011

Configuration
-------------
Example:

.. code-block:: yaml

    phenology:
        enable: False
        # Specification for dataset indices required for EVI based phenology monitoring
        red_index: 2
        nir_index: 3
        blue_index: 0
        # Scale factor for reflectance bands
        scale: 0.0001
        # You can also specify index of EVI if contained in dataset to override calculation
        evi_index:
        evi_scale:
        # Number of years to group together when normalizing EVI to upper and lower percentiles
        year_interval: 3
        # Upper and lower percentiles of EVI used for max/min scaling
        q_min: 10
        q_max: 90
