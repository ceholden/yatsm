# Regression Models

This directory contains example regression method objects from `scikit-learn`. These objects are serialized to disk into Python "pickle" files created using `sklearn.external.joblib`. These "pickled" files are included as examples because `YATSM` (`YATSM>=0.5.0`) can be run using a variety of prediction methods as long as they are serializable as a class object with a `fit` and `predict` interface similar to estimators from the `scikit-learn` package.

## Examples
Current examples include:

1. `Lasso_alpha20.pkl`
    - Lasso regression method where `alpha` is fixed to a value of `20`. This specific parameterization of Lasso regression is used by Zhu Zhe in the CCDC algorithm.
2. `LassoCV_n100_alpha_0-50.pkl`
    - Lasso regression method where the `alpha` (usually called `lambda`, the tradeoff between least squares and L1 shrinkage) hyperparameter is crossvalidated among `n=100` values ranging between `1e-3` and `50`.
3. `OLS.pkl`
    * Ordinary Least Squares
4. `glmnet_LassoCV_n50_lambda_0-35.pkl`
    - Lasso regression with the `lambda` (`alpha` in `scikit-learn`) chosen by cross-validation among `n=50` values ranging between `1e-4` and `35` (spacing via `np.logspace`)

## Creation

Custom regression estimators may be created as "pickles" as follows:

``` python
In [1]: import sklearn.linear_model, sklearn.externals

In [2]: lasso = sklearn.linear_model.Lasso(alpha=20.0)

In [3]: sklearn.externals.joblib.dump(lasso, 'Lasso_alpha20.pkl', compress=3)
Out[3]: ['Lasso_alpha20.pkl']
```

Note that this example uses compression to group all pickled files into one compressed file. If compression is not used and the regression estimator has a NumPy array as an attribute, `joblib` will create separate pickle files for the regression object and the NumPy array attribute.
