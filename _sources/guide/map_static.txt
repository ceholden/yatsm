.. _guide_map_static:

===========================
Mapping Derived Information
===========================

Maps can be created based on the attributes of a timeseries segment for any date desired. These attributes include the statistical model coefficients for each band, the Root Mean Squared Error (RMSE) of the model fits for each band, and the classification value predicted. The predicted reflectance for any band may also be generated using the model coefficients. These maps may be created using the :ref:`yatsm map <yatsm_map>` script.

Missing Values
==============

By default, :ref:`yatsm map <yatsm_map>` will only look for models which intersect the date given by the user. A model intersects a date if the model's starting date is equal to or less than the date and if the model's ending date is equal to or greater than the date. For example, a model beginning on 2000-01-01 and ending on 2002-01-01 intersects 2001-01-01, but this model does not intersect 2002-02-02.

Sometimes it is desirable to produce a classification value or reflectance value even if no models intersect the date provided. For example, we wanted to produce a classification value for 2000-01-01 but the model ended in 1999 and was not re-initialized until 2000-06-01, we might be happy to know what the classification value of the next segment is (i.e., what the pixel turns into).

In other circumstances, we might be okay producing a map using the segment immediately prior to the date of the map provided that there was not a change detected. This is a common case when generating classifications, coefficient maps, or reflectance predictions toward the end of the timeseries.

For these two circumstances, the :ref:`yatsm map <yatsm_map>` script provides the ``--before`` and ``--after`` flags. These flags are not mutually exclusive and will not override the default behavior of providing the timeseries segment information which intersects the date provided. Instead, these three choices operate in order of desirability. From most to least desirable:

1. Segment which intersects the date
2. Segment immediately after the date, if ``-after`` is specified
3. Segment before the date, if the segment does not contain a break and ``--before`` is specified

The mapped output will always contain the information from the most desirable segment. Behind the scenes, it accomplishes this by providing mapped values for the least desirable option first and then overwriting with more desirable options.

Examples
========

For example, consider these mapping situations:

1. Create a map showing the image predicted for January 1st, 2000 for all bands::

.. code-block:: bash

    yatsm line predict 2000-01-01 predict_2000-01-01.gtif

2. Create a map showing the image predicted for January 1st, 2000 for all bands, filling in missing values with the information from the timeseries segment immediately after January 1st, 2000, if possible::

.. code-block:: bash

    yatsm map --after predict 2000-01-01 predict_2000-01-01.gtif

3. Create a map showing all the coefficients for the red band (band 3) for January 1st, 2000::

.. code-block:: bash

    yatsm map --band 3 coef 2000-01-01 coef_red_2000-01-01.gtif

4. Create a map of only the time trend, or slope, coefficients for all bands for January 1st, 2000::

.. code-block:: bash

    yatsm map --coef slope coef 2000-01-01 coef_slope_2000-01-01.gtif

5. Create a map of the current land cover, or the land cover that a pixel will turn into, for January 1st, 2000::

.. code-block:: bash

    yatsm map --after class 2000-01-01 classmap_2000-01-01.gtif

Docs TODO
=========

- Example maps
- Images helping explain ``--after`` and ``--before``
- More information on CLI flags / switches
- Explanation of ``--root``, ``--result``, and ``--image`` parameters
