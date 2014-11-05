Batch Process Configuration
===========================

## Configuration File
The batch running script uses an [INI file](https://wiki.python.org/moin/ConfigParserExamples) to parameterize the run. The INI file uses two sections - one describing the dataset and another detailing the model parameters.

## Dataset Parameters

The following dataset information is required:

| Parameter | Data Type | Explanation |
| :-------: | :-------: | :---------: |
| `input_file` | `filename` | The filename of a CSV file recording the date and filenames of all images in the dataset |
| `date_format` | `str` | The format of the dates specified in the `input_file` (e.g., `%Y%j`) |
| `output` | `str` | Output folder location for results |
| `n_bands` | `int` | The number of bands in the images |
| `mask_band` | `int` | Band index in each image of the mask band |
| `green_band` | `int` | Band index in each image of the green band (~520-600 nm) |
| `swir1_band` | `int` | Band index in each image of the shortwave-infrared band (~1550-1750 nm) |
| `use_bip_reader` | `bool` | Use `fopen` style read in for band interleave by pixel (BIP) files, instead of GDAL's IO |

**Note**: you can use `scripts/gen_date_file.sh` to generate the CSV file for `input_file`.

## Model Parameters

The change detection is parameterized by:

| Parameter | Data Type | Explanation |
| :-------: | :-------: | :---------: |
| `consecutive` | `int` | Consecutive observations to trigger change |
| `threshold` | `float` | Test statistic critical value to trigger change |
| `min_obs` | `int` | Minimum observations per time segment in model |
| `min_rmse` | `float` | Minimum RMSE in test statistic for each model |
| `freq` | `list` | Frequency of sine/cosine seasonal periods |
| `test_indices` | `list` | Indices of Y to use in change detection |
| `screening` | `str` | Method for screening timeseries for noise. Options are [`RLM`](http://statsmodels.sourceforge.net/stable/generated/statsmodels.robust.robust_linear_model.RLM.html) and [`LOWESS`](http://statsmodels.sourceforge.net/stable/generated/statsmodels.nonparametric.smoothers_lowess.lowess.html?highlight=lowess#statsmodels.nonparametric.smoothers_lowess.lowess) |
| `lassocv` | `bool` | Use [`sklearn.linear_model.LassoLarsCV`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV) instead of [`glmnet`](https://github.com/dwf/glmnet-python) |
| `reverse` | `bool` | Run model backward in time, rather than forward |
| `robust` | `bool` | Return coefficients and RMSE from a robust linear model for each time segment |

## Example

An example template of the parameter file is located within `examples/example.ini`.
