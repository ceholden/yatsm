# Change Log

All notable changes will appear in this log that begins with the release of
`v0.5.0`. Changes are categorized into "Added", "Changed", "Fixed", and "Removed". To see a comparison between releases on Github, click or follow the release version number URL.

For information on the style of this change log, see [keepachangelog.com](http://keepachangelog.com/).

## [UNRELEASED (`v0.6.0`)](https://github.com/ceholden/yatsm/compare/v0.5.5...HEAD)

[Milestone v0.6.0](https://github.com/ceholden/yatsm/milestones/v0.6.0)

### Changed
- `CCDCesque`: Optimize algorithm implementation. Performance estimates show 2x speed gain [#70](https://github.com/ceholden/yatsm/issues/70)
- CLI: Improve `yatsm pixel` `--embed` option ([commit](https://github.com/ceholden/yatsm/commit/b1cf47ff3feeeb93b9f671bccc4379a9da1ad808))
- CLI: Add `--verbose-yatsm` to main `yatsm` command so it works with all programs running a YATSM algorithm ([commit](https://github.com/ceholden/yatsm/commit/772badc980c56d2d5c4185a40bf856bc6875be91))

### Added
- Add submodule `yatsm.regression.diagnostics` for regression diagostics, including RMSE ([commit](https://github.com/ceholden/yatsm/commit/df582d235a6e6c8e114053015a7b7392bee8f570))
- Add new module `yatsm.accel` with decorator (`try_jit`) that applies `numba.jit` to functions only if `numba` is available [#70](https://github.com/ceholden/yatsm/issues/70)
- Apply `yatsm.accel.try_jit` to calculation of `yatsm.regression.diagnostics.rmse`, `yatsm.regression.robust_fit.RLM`, and others [#70](https://github.com/ceholden/yatsm/issues/70)
- asv
- Improve `clean` target in package's `setup.py` so it deletes built estimator pickles and `.c`/`.so` built with Cython ([commit](https://github.com/ceholden/yatsm/commit/bb868922a2f6f2f68c9f71153c4307e8727468cb))

### Fixed
- `CCDCesque`: Fix bug in calculation of `end` attribute for last timeseries record [#72](https://github.com/ceholden/yatsm/issues/72)
- `CCDCesque`: Fix bug in parsing of `test_indices` if user doesn't supply any [#73](https://github.com/ceholden/yatsm/issues/73)
- "Packaged" estimator pickles are built on installation of YATSM so they will work with user versions of libraries ([commit](https://github.com/ceholden/yatsm/commit/d9b4b80c1c70137525abfde7fc7933e34bcf6820))
- Fix `DeprecationWarnings` with `scikit-learn>=0.17.0` ([commit](https://github.com/ceholden/yatsm/commit/29ddd4c0da29904b49fca7e452ee23ca1f938261))

## [v0.5.5](https://github.com/ceholden/yatsm/compare/v0.5.4...v0.5.5) - 2015-11-24

[Milestone v0.5.5](https://github.com/ceholden/yatsm/milestones/v0.5.5)

### Added
- Abort if config file 'n_bands' looks incorrect ([commit](https://github.com/ceholden/yatsm/commit/01a6adec1fcd567c194e28b98fa488c13cdbdd45))

### Changed
- Reorganize long term mean phenology code into generic phenology related submodule.
- Reorganize changemap and map logic to separate module [#60](https://github.com/ceholden/yatsm/issues/60)

### Fixed
- Fix bug with spline EVI prediction in LTM phenology module when data include last day in leap year (366) [#56](https://github.com/ceholden/yatsm/issues/56)
- Fix bug with phenology half-max calculation that created erroneous transition dates [#58](https://github.com/ceholden/yatsm/issues/58)
- Fix bug with phenology calculation for 100% masked data pixels [#54](https://github.com/ceholden/yatsm/issues/54)
- Fix `yatsm pixel` to correctly plot designs that include categorical variables ([commit](https://github.com/ceholden/yatsm/commit/966edd8b4a95e3c19d677eb71e2b76a155911d88))
- Fix passing of a list of dataset min/max values within config files instead of 1 number [#59](https://github.com/ceholden/yatsm/issues/59)
- Add missing `phenology` module to `setup.py` ([commit](https://github.com/ceholden/yatsm/commit/9d49d737316b34d2465b18db55647d7104d17758))


## [v0.5.4](https://github.com/ceholden/yatsm/compare/v0.5.3...v0.5.4) - 2015-10-28

[Milestone v0.5.4](https://github.com/ceholden/yatsm/milestones/v0.5.4)

### Fixed
- Fix multiple bugs encountered when running phenology estimates [#49](https://github.com/ceholden/yatsm/issues/49)

### Changed
- Metadata from `yatsm line` runs are now stored in `metadata` sub-file of NumPy compressed saved files [#53](https://github.com/ceholden/yatsm/issues/53)
- Algorithm configurations must now declare subsections that match estimator methods (e.g., `init` and `fit`) [#52](https://github.com/ceholden/yatsm/issues/52)
- Refactored `yatsm.phenology` to make `LongTermMeanPhenology` estimator follow `scikit-learn` API [#50](https://github.com/ceholden/yatsm/issues/50)

### Added
- Add `--num_threads` option to `yatsm` CLI. This argument sets various environment variables (e.g., `OPENBLAS_NUM_THREADS` or `MKL_NUM_THREADS`) before beginning computation to set or limit multithreaded linear algebra calculations within NumPy [#51](https://github.com/ceholden/yatsm/issues/51)
- Add this changelog!

## [v0.5.3](https://github.com/ceholden/yatsm/compare/v0.5.2...v0.5.3) - 2015-10-20

[Milestone v0.5.3](https://github.com/ceholden/yatsm/milestones/v0.5.3)

### Changed
- Update configuration file parsing for classifiers to use `yaml`. Users need to update their classifier configuration files to use this new format.

### Fixed
- Fixed bug when running on real datasets with 100% missing data in timeseries (e.g., in scene corners) [#47](https://github.com/ceholden/yatsm/issues/47) [#48](https://github.com/ceholden/yatsm/issues/48)
- Fix `yatsm train` and `yatsm classify` for `v0.5.0+` releases

### Removed
- Deleted intermediate "helper" classes that were used to type-check `ini` configuration files

## [v0.5.2](https://github.com/ceholden/yatsm/compare/v0.5.1...v0.5.2) - 2015-10-09

[Milestone v0.5.2](https://github.com/ceholden/yatsm/milestones/v0.5.2)

### Fixed
- Catch `TSLengthException` so `yatsm line` can continue running [#43](https://github.com/ceholden/yatsm/issues/43)
- Allow refit estimators to be from pre-packaged, distributed pickles [#44](https://github.com/ceholden/yatsm/issues/44)
- Remove references to old variable names in `yatsm.algorithms.postprocess` [#45](https://github.com/ceholden/yatsm/issues/45)

## [v0.5.1](https://github.com/ceholden/yatsm/compare/v0.5.0...v0.5.1) - 2015-10-06

[Milestone v0.5.1](https://github.com/ceholden/yatsm/milestones/v0.5.1)

### Added
- Use environment variables in configuration files [#42](https://github.com/ceholden/yatsm/issues/42)
- Pre-package a set of pickled estimators using `package_data` from `setuptools` [#41](https://github.com/ceholden/yatsm/issues/41)

## v0.5.0 - 2015-09-14

[Milestone v0.5.0](https://github.com/ceholden/yatsm/milestones/v0.5.0)

Very backwards incompatible release required to redefine project objectives and
use better technology (click & YAML) for command line interface.

### Changed
- Command line interface uses [`click`][click.pocoo.org] [#28](https://github.com/ceholden/yatsm/issues/28)
- Redefine `YATSM` as baseclass and rename CCDC implementation to `CCDCesque` [#29](https://github.com/ceholden/yatsm/issues/28)
- Specify prediction method using serialized "pickle" instances of `scikit-learn` compatible estimators [#26](https://github.com/ceholden/yatsm/issues/26)
- Configuration file now uses `YAML` format for better organization and more sustainable parsing [#30](https://github.com/ceholden/yatsm/issues/30)
- Refactor `robust` fit into more generalized `refit` step. User can generate additional `[prefix]_coef` and `[prefix]_rmse` results using specified estimators [#33](https://github.com/ceholden/yatsm/issues/33)
- Tests now use `py.test` fixtures for better code reuse
- Reorganize `requirements.txt` organization and documentation

### Added
- Add `environment.yaml` for creating environments within the [Anaconda](https://www.continuum.io/downloads) distribution using `conda`
