Robust Linear Model Benchmark
-----------------------------

[Statsmodels](http://statsmodels.sourceforge.net/) contains a very well written routine to calculate a robust linear model using iteratively reweighted least squares. While this routine, [`statsmodels.robust.robust_linear_model.RLM`](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/robust/robust_linear_model.py) is feature rich, it costs too much time for our purposes considering we don't need anything more than model fit and prediction capability. To increase speed, `yatsm.regression.robust_fit.RLM` was created as a carbon-copy with all the superfluous features removed.

## Benchmark

Using `statsmodels.robust.robust_linear_model.RLM` implementation:

    > python -m cProfile -o sandbox/benchmark/robustLM/line_sm.prof ./scripts/line_yatsm.py -v examples/example.ini 1 125

Using `yatsm.regression.robust_fit.RLM` implementation:

    > python -m cProfile -o sandbox/benchmark/robustLM/line_rlm.prof ./scripts/line_yatsm.py -v examples/example.ini 1 125

## Comparison

### `statsmodels.robust.robust_linear_model.RLM`

    line_sm.prof% sort tottime
    line_sm.prof% stats 10
    Fri Feb 20 15:30:03 2015    line_sm.prof

             52346153 function calls (50185897 primitive calls) in 113.119 seconds

       Ordered by: internal time
       List reduced from 4517 to 10 due to restriction <10>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      1851858   10.847    0.000   10.847    0.000 {method 'reduce' of 'numpy.ufunc' objects}
       312452    8.627    0.000   17.697    0.000 /usr/local/lib/python2.7/dist-packages/scipy/linalg/decomp_svd.py:15(svd)
       156226    8.311    0.000   10.671    0.000 /usr/local/lib/python2.7/dist-packages/numpy/linalg/linalg.py:1225(svd)
    1717932/858966    4.743    0.000   15.966    0.000 /usr/local/lib/python2.7/dist-packages/statsmodels/base/wrapper.py:22(__getattribute__)
       312452    4.127    0.000   27.185    0.000 /usr/local/lib/python2.7/dist-packages/statsmodels/tools/tools.py:374(rank)
       156226    4.026    0.000    7.526    0.000 /usr/local/lib/python2.7/dist-packages/numpy/core/_methods.py:77(_var)
      3175902    3.117    0.000    3.117    0.000 {numpy.core.multiarray.array}
        15678    3.091    0.000  101.084    0.006 /usr/local/lib/python2.7/dist-packages/statsmodels/robust/robust_linear_model.py:198(fit)
       140548    2.945    0.000    3.984    0.000 /usr/local/lib/python2.7/dist-packages/statsmodels/robust/norms.py:726(rho)
      1129954    2.849    0.000    2.849    0.000 {numpy.core.multiarray.dot}


### `yatsm.regression.robust_fit.RLM`
    
    line_rlm.prof% sort tottime
    line_rlm.prof% stats 10
    Fri Feb 20 15:37:18 2015    line_rlm.prof

             17382616 function calls (17377614 primitive calls) in 39.607 seconds

       Ordered by: internal time
       List reduced from 4456 to 10 due to restriction <10>

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       284396    5.739    0.000    5.739    0.000 {numpy.linalg.lapack_lite.dgelsd}
       142198    4.817    0.000   18.635    0.000 /usr/local/lib/python2.7/dist-packages/numpy/linalg/linalg.py:1733(lstsq)
       592964    3.162    0.000    3.162    0.000 {method 'reduce' of 'numpy.ufunc' objects}
       142198    2.642    0.000   22.464    0.000 yatsm/regression/robust_fit.py:60(_weight_fit)
      1789734    2.236    0.000    2.236    0.000 {numpy.core.multiarray.array}
       126510    1.649    0.000    1.649    0.000 yatsm/regression/robust_fit.py:18(bisquare)
        15688    1.395    0.000   34.527    0.002 yatsm/regression/robust_fit.py:119(fit)
       126510    1.103    0.000    2.374    0.000 yatsm/regression/robust_fit.py:56(_check_converge)
       142198    1.001    0.000    5.027    0.000 /usr/local/lib/python2.7/dist-packages/numpy/lib/function_base.py:2896(_median)
         5196    0.882    0.000    1.463    0.000 yatsm/yatsm.py:648(monitor)


## Conclusion

The switch sped up the total runtime by just about 300%, with the `fit` methods of each implementation taking up 34.527s for `yatsm.regression.robust_fit.RLM` and 101.084s for `statsmodels.robust.robust_linear_model.RLM`.
