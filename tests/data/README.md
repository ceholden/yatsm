# YATSM Test Data Files

1. `p035r032_testdata.tar.gz`
    * Modification of example timeseries stack for `p035r032` (see [landsat_stack repo](https://github.com/ceholden/landsat_stack))
        * Original dataset reshaped from first 25 pixels in first row to 5x5
        * Modified to include varying levels of NODATA
    * Data generating process: `yatsm/sandbox/test_data/create_data.py`
2. `cache/yatsm_r0_n447_b8.npy.npz`
    * First line of example timeseries from `p035r032` subset timeseries (see #1)
3. `p013r030_r50_n423_b8.npz`
    * NumPy compressed data for row 50 of `p013r030` subset
    * Contains:
        * `Y`: observed row of data
        * `dates`: ordinal dates for each observations in `Y`
        * `X`: design matrix
        * `design_str`: `"1 + x + harm(x, 1) + harm(x, 2)"`
        * `design_dict`: `OrderedDict` containing design info from `patsy`
4. `results`
    * Results from `p035r032_testdata.tar.gz` ran with `results/p035r032_results.yaml`
    * `results/YATSM` results
    * `results/YATSM_classified` results, with arbitrary classification
        * Note that these results have a "class" and "class_proba" attribute added with arbitrary values for testing purposes
        * Data generating process `results/add_arbitrary_classification.py`
    * `results/cache` cached data
    * `results/example_image.gtif` example image from timeseries
