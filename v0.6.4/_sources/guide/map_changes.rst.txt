.. _guide_map_changes:

===============
Mapping Changes
===============

To visualize change information it is often useful to create maps of the date when a pixel changes or to map the total number of changes detected within a desired range of dates. These maps can be created using the :ref:`yatsm changemap <yatsm_changemap>`.

Examples
========

For example, consider these mapping situations:

1. For each year from 2000 to 2010, create a map of the first change detected within the year::

    for y1 in $(seq 2000 2010); do
        y2=$(expr $y1 + 1)
        yatsm changemap first $y1-01-01 $y2-01-01 change_$y1-$y2.gtif
    done

2. Create a map of the total number of changes, per pixel, for the 2000 to 2010 decade::

    yatsm changemap num 2000-01-01 2010-01-01 changenumber_2000-2010.gtif

Docs TODO
=========

- Example images
- Explanation of ``--root``, ``--result``, and ``--image`` parameters
