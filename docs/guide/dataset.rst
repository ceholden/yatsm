.. _guide_dataset_prep:

===================
Dataset Preparation
===================

This part of the guide is a work in progress. For now you can get an idea of how
the datasets should be organized and prepared by viewing two example datasets
located [here](https://github.com/ceholden/landsat_stack).

# Mask Bands
Mask bands (Fmask, etc.) currently must be the last band in the image stack. It is possible to modify the code such that the mask band can be in any order, but this is a low priority feature.
