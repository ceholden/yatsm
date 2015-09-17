# Yet Another Timeseries Model (YATSM)

[![Join the chat at https://gitter.im/ceholden/yatsm](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/ceholden/yatsm?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Build Status](https://travis-ci.org/ceholden/yatsm.svg)](https://travis-ci.org/ceholden/yatsm) [![Coverage Status](https://coveralls.io/repos/ceholden/yatsm/badge.svg?branch=v0.4.0)](https://coveralls.io/r/ceholden/yatsm?branch=v0.4.0) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.17129.svg)](http://dx.doi.org/10.5281/zenodo.17129)

## About
The Yet Another TimeSeries Model (YATSM) algorithm is designed to monitor land
surface phenomena, including land cover and land use change, using timeseries
of remote sensing observations. The algorithm seeks to find distinct time
periods, or time segments, within the timeseries by monitoring for disturbances. These time segments may be used to infer continuous periods of stable land cover, with breaks separating the segments representing ephemeral disturbances or permanent conversions in land cover or land use.

The ["Yet Another..."](http://en.wikipedia.org/wiki/Yet_another) part of the algorithm name is an acknowledgement of the influence a previously published timeseries algorithm - the Continuous Change Detection and Classification (CCDC) (Zhu and Woodcock, 2014) algorithm. While YATSM began as an extension from CCDC, it was never intended as a 1 to 1 port of CCDC and will continue to diverge in its own direction.

This algorithm is also influenced by other remote sensing algorithms which, like CCDC, are based in theory on tests for structural change from econometrics
literature (Chow, 1960; Andrews, 1993; Chu *et al*, 1996; Zeileis, 2005). These other remote sensing algorithms include Break detection For Additive Season and Trend (BFAST) (Verbesselt *et al*, 2012) and LandTrendr (Kennedy *et al*, 2010). By combining ideas from CCDC, BFAST, and LandTrendr, this "Yet Another..." algorithm hopes to overcome weaknesses present in any single approach.

Please consider citing as:

    Christopher E. Holden. (2015). Yet Another Time Series Model (YATSM). Zenodo. 10.5281/zenodo.17129

## Documentation

Documentation is available [here](http://ceholden.github.io/yatsm/).

Contributions to the documentation, especially for the user guide, is more than welcomed. The documentation for this project is built using [Sphinx](http://sphinx-doc.org/) using the [ReadTheDocs](https://readthedocs.org/) theme. See the `docs/` folder for documentation source material.

## Example
The simplest way of using YATSM would be the pixel-by-pixel command line interface - `run_yatsm.py`.

We'll use the example [Landsat stack from Chiapas, Mexico](https://github.com/ceholden/landsat_stack) for this demonstration:

``` bash
    > yatsm pixel --band 2 --style xkcd examples/p022r049/p022r049.yaml 133 106
```

Produces:
    ![Timeseries](docs/media/double_cut_ts_b3.png)
    ![Modeled Timeseries](docs/media/double_cut_ts_fitted_b3.png)

## Installation

It is strongly encouraged that you install YATSM into an isolated environment, either using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) for `pip` installs or a separate environment using [`conda`](http://conda.pydata.org/docs/), to avoid dependency conflicts with other software.

This package requires an installation of [`GDAL`](http://gdal.org/), including the Python bindings. Note that [`GDAL`](http://gdal.org/) version 2.0 is not yet tested (it probably works, but I haven't tried GDAL 2.x), but recent 1.x versions (likely 1.8+) should work. [`GDAL`](http://gdal.org/) is not installable solely via `pip` and needs to be installed prior to following the `pip` instructions. If you follow the instructions for [`conda`](http://conda.pydata.org/docs/), you will not need to install `GDAL` on your own because [`conda`](http://conda.pydata.org/docs/) packages a compiled copy of the `GDAL` library (yet another reason to use [`conda`](http://conda.pydata.org/docs/)!).

### pip
The basic dependencies for YATSM are included in the `requirements.txt` file which is  by PIP as follows:

``` bash
    pip install -r requirements.txt
```

Additional dependencies are required for some timeseries analysis algorithms or for accelerating the computation in YATSM. These requirements are separate from the common base installation requirements so that YATSM may be more modular:

* Long term mean phenological calculations from Melaas *et al.*, 2013
    * Requires the R statistical software environment and the `rpy2` Python to R interface
    * `pip install -r requirements/pheno.txt`
* Computation acceleration
    * GLMNET Fortran wrapper for accelerating Elastic Net or Lasso regularized regression
    * Numba for speeding up computation through just in time compilation (JIT)
    * `pip install -r requirements/accel.txt`

A complete installation of YATSM, including acceleration dependencies and additional timeseries analysis dependencies, may be installed using the `requirements/all.txt` file:

``` bash
    pip install -r requirements/all.txt
```

### Conda
Requirements for YATSM may also be installed using [`conda`](http://conda.pydata.org/docs/), Python's cross-platform and platform agnostic binary package manager from [ContinuumIO](http://continuum.io/). [`conda`](http://conda.pydata.org/docs/) makes installation of Python packages, especially scientific packages, a breeze because it includes compiled library dependencies that remove the need for a compiler or pre-installed libraries.

Installation instructions for `conda` are available on their docs site [conda.pydata.org](http://conda.pydata.org/docs/get-started.html)

Since [`conda`](http://conda.pydata.org/docs/) makes installation so easy, installation through [`conda`](http://conda.pydata.org/docs/) will install all non-developer dependencies. Install YATSM using [`conda`](http://conda.pydata.org/docs/) into an isolated environment by using the `environment.yaml` file as follows:

``` bash
    # Install
    conda env create -n yatsm -f environment.yaml
    # Activate
    source activate yatsm
```
