# YATSM Test Data Files

1. `p035r032_subset.tar.gz`
    * Example timeseries stack for `p035r032` (see [landsat_stack repo](https://github.com/ceholden/landsat_stack))
    * Zipped tarball
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
