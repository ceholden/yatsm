Yet Another Timeseries Model (YATSM)
------------------------------------
## About
YATSM model based off of tests for structural changes from the econometrics literature including the MOSUM or CUMSUM (Chu et al, Zeileis, and others) as implemented in a remote sensing context by BFAST (Verbesselt, et al. 2012) and CCDC (Zhu and Woodcock, 2014). This effort is not intended as a direct port of either algorithm. The "YATSM" name intends to refer to these two algorithms without claiming 1 to 1 functionality of either.

Documentation on how to use the scripts and utilities built for the model can be found [in the `docs` folder](docs/README.md).

## Example
The simplest way of using YATSM would be the pixel-by-pixel command line interface - `run_yatsm.py`.

We'll use the example [Landsat stack from Chiapas, Mexico](https://github.com/ceholden/landsat_stack) for this demonstration:

    > run_yatsm.py --consecutive=5 --threshold=3 --min_obs=16 \
    ... --freq="1, 2" --min_rmse 150 --test_indices "2 4 5" --screening RLM \
    ... --plot_index=4 --plot_style xkcd \
    ... ../landsat_stack/p022r049/images/ 150 50

Produces:
    ![Timeseries](docs/media/example_ts_b5.png)
    ![Modeled Timeseries](docs/media/example_ts_fitted_b5.png)

## Requirements
#### Main dependencies:

    Python (2.7.x tested)
    GDAL (1.10.0 tested)

#### Python dependencies:
Listed below are the Python library requirements for running YATSM. The version numbers listed are the versions I've used for development, but I suspect the versions are flexible.

    cython >= 0.20.1
    numpy >= 1.8.1
    pandas >= 0.13.1
    statsmodels >= 0.5.0
    glmnet = 1.1-5 (see: https://github.com/dwf/glmnet-python)
    scikit-learn >= 0.15.1
    ggplot >= 0.5.8
    docopt >= 0.6.1
