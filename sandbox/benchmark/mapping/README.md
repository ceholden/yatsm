Prediction / Coefficient Mapping
================================

## Description
Previously the code that generated prediction or coefficient images processed all valid model results (that is, model results that started before and ended after a specified date) inside a loop. Looping made the arithmetic (dot products, etc.) much easier to accomplish. However, some more advanced indexing from NumPy now allows for the elimination of this loop, creating gigantic speed-ups.

## Performance Benefit

The `timeit` module from the Python standard library was used to test the performance increase associated with the changes (see [test_prediction.py](./test_prediction.py)). Performance was greatly increased while creating more readible, compact code:

    $ ./test_prediction.py 
    PREDICTION:
    Loop version:
    5.59993505478
    numpy.tensordot
    0.12917804718
    tensordot speed: 4335.052% faster
    
    COEFICIENT:
    Loop version:
    5.02235102654
    numpy.dot
    0.354978084564
    dot product speed: 1414.834% faster
    
    NumPy version: 1.8.2

## Technical Description

A diff example from the coefficient mapping section:

    diff --git a/scripts/yatsm_map.py b/scripts/yatsm_map.py
    index 65ea3b7..87b9fc2 100755
    --- a/scripts/yatsm_map.py
    +++ b/scripts/yatsm_map.py
    @@ -321,19 +321,18 @@ def get_coefficients(date, bands, coefs, results, image_ds,
             if index.shape[0] == 0:
                 continue
     
    -        for i in index:
    -            # Normalize intercept to mid-point in time segment
    -            rec['coef'][i][0, :] = rec['coef'][i][0, :] + \
    -                (rec['start'][i] + rec['end'][i]) / 2.0 * rec['coef'][i][1, :]
    +        # Normalize intercept to mid-point in time segment
    +        rec['coef'][i][0, :] += \
    +            ((rec['start'][index] + rec['end'][index]) / 2.0)[:, None] * \
    +            rec['coef'][index, 1, :]
     
    -            # Extract coefficients
    -            raster[rec['py'][i], rec['px'][i], range(n_coefs * n_bands)] = \
    -                rec['coef'][i][i_coefs, :][:, i_bands].flatten()
    +        # Extract coefficients
    +        raster[rec['py'][i], rec['px'][i], range(n_coefs * n_bands)] = \
    +            rec['coef'][index][:, coefs, :][:, :, bands]
     
    -            # Extract RMSE
    -            if use_rmse is True:
    -                raster[rec['py'][i], rec['px'][i], n_coefs * n_bands:] = \
    -                    rec['rmse'][i][i_bands]
    +        if use_rmse:
    +            raster[rec['py'][i], rec['px'][i], n_coefs * n_bands:] = \
    +                rec['rmse'][i][i_bands]
     
         return (raster, band_names)


