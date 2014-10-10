Multi-temporal noise screening benchmark
========================================

# Approaches:
1. Robust Linear Models (statsmodels.api.RLM)
2. LOWESS (statsmodels.nonparametric.lowess)

# Background:
We want to detect outliers (clouds, shadows, haze, etc.) in our dataset, especially when we are going to train a prediction model. To test if a given observation is noisy, we need to know what the observation should look like. We assert an observation is noise if it the residual from our prediction is large enough in certain directions using the SWIR1 and Green bands.

### Robust Linear Models
This approach does iterative re-weighted least-squares regression that penalizes at each iteration observations which have large residuals. When the iteration stops, we should have a regression model which accurately predicts how most of the observations look - ignoring noise.

Because fit one global model, we limit the screening analysis to the training period.

### LOWESS
This approach uses locally weighted regression within a fixed "span" window to smooth the data. Because the regressions are localized, the LOWESS analysis can be performed across the entire timeseries. One issue, however, is that our data are not necessarily equally spaced in time. Thus, our fixed "span" is only fixed with respect to the available observations, not actual time.

# Benchmark

The benchmark was performed by running YATSM for an entire line using the built-in `example.ini`.

First, set `screening = LOWESS` in `example.ini`. Next:

    # LOWESS
    > python -m cProfile -o lowess.prof yatsm/line_yatsm.py -v examples/example.ini 1 250

Again, reset `screening = RLM` in  `example.ini`. Next:

    # RLM
    > python -m cProfile -o rlm.prof yatsm/line_yatsm.py -v examples/example.ini 1 250

We can analyze the `lowess.prof` and `rlm.prof` using `pstats` module. Some output sorted by cumulative time (`cumtime`):

### LOWESS
    Thu Sep  4 15:27:05 2014    lowess.prof

         5873229 function calls (5869159 primitive calls) in 35.101 seconds

    Ordered by: cumulative time
    List reduced from 3933 to 20 due to restriction <20>

    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.002    0.002   35.114   35.114 line_yatsm.py:14(<module>)
        1    0.000    0.000   34.498   34.498 line_yatsm.py:304(main)
        1    0.067    0.067   34.482   34.482 line_yatsm.py:202(run_line)
      250    0.033    0.000   34.243    0.137 line_yatsm.py:263(run_pixel)
      250    0.110    0.000   34.200    0.137 yatsm.py:450(run)
    18323    0.251    0.000   16.335    0.001 yatsm.py:548(train)
     1379    0.016    0.000   14.752    0.011 yatsm.py:475(screen_timeseries_LOWESS)
      250    0.011    0.000   14.736    0.059 yatsm.py:148(smooth_mask)
      500    0.014    0.000   14.720    0.029 smoothers_lowess.py:13(lowess)
      500   12.323    0.025   14.692    0.029 {statsmodels.nonparametric._smoothers_lowess.lowess}
    24098    9.512    0.000   14.062    0.001 yatsm.py:610(monitor)
     5100    0.736    0.000    4.439    0.001 yatsm.py:638(fit_models_GLMnet)
    898436    2.572    0.000    3.803    0.000 elastic_net.py:23(predict)
    24098    0.332    0.000    3.657    0.000 yatsm.py:587(update_model)
    35700    1.199    0.000    3.519    0.000 yatsm.py:27(fit)
    278780    0.486    0.000    2.687    0.000 fromnumeric.py:1621(sum)
    335885    2.032    0.000    2.032    0.000 {method 'reduce' of 'numpy.ufunc' objects}
    278780    0.292    0.000    1.989    0.000 _methods.py:23(_sum)
    35700    0.734    0.000    1.451    0.000 glmnet.py:8(elastic_net)
    898436    1.232    0.000    1.232    0.000 {numpy.core._dotblas.dot}

### RLM
    Thu Sep  4 15:28:32 2014    rlm.prof

             39556053 function calls (38122003 primitive calls) in 101.116 seconds

       Ordered by: cumulative time
       List reduced from 4008 to 20 due to restriction <20>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.002    0.002  101.129  101.129 line_yatsm.py:14(<module>)
            1    0.000    0.000  100.517  100.517 line_yatsm.py:304(main)
            1    0.066    0.066  100.501  100.501 line_yatsm.py:202(run_line)
          250    0.035    0.000  100.266    0.401 line_yatsm.py:263(run_pixel)
          250    0.215    0.001  100.222    0.401 yatsm.py:450(run)
        13648    0.265    0.000   73.867    0.005 yatsm.py:548(train)
         4945    0.319    0.000   72.188    0.015 yatsm.py:502(screen_timeseries_RLM)
         4945    0.436    0.000   71.737    0.015 {yatsm.cyatsm.multitemp_mask}
         9890    2.036    0.000   67.190    0.007 robust_linear_model.py:198(fit)
        93354    0.515    0.000   28.646    0.000 linear_model.py:375(__init__)
        93354    0.302    0.000   27.942    0.000 linear_model.py:78(__init__)
        93354    0.228    0.000   27.586    0.000 model.py:135(__init__)
        35355   15.354    0.000   21.857    0.001 yatsm.py:610(monitor)
        93354    0.704    0.000   18.886    0.000 linear_model.py:82(initialize)
       206488    2.751    0.000   17.881    0.000 tools.py:374(rank)
        93354    0.608    0.000   12.306    0.000 robust_linear_model.py:173(_update_history)
        93354    0.757    0.000   11.932    0.000 linear_model.py:93(fit)
       206488    0.239    0.000   11.589    0.000 decomp_svd.py:113(svdvals)
       206488    5.230    0.000   11.350    0.000 decomp_svd.py:15(svd)
       103244    2.136    0.000   10.389    0.000 linalg.py:1519(pinv)

# Results:
Not surprisingly, performing the noise removal only once rather than at least once per training period is much faster.

The output results differ, however. The two methods are not equivalent. From looking at the profile, it seems that the LOWESS screening results in more breaks being found:

    # LOWESS
    18323 calls to yatsm.py:548(train))
    # RLM
    13648 calls to yatsm.py:548(train))

