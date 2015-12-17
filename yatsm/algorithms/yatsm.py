""" Yet Another TimeSeries Model baseclass
"""
import numpy as np
import patsy
import sklearn
import sklearn.linear_model

from .. import _cyprep as cyprep
from ..regression.diagnostics import rmse
from ..regression.transforms import harm  # noqa


class YATSM(object):
    """ Yet Another TimeSeries Model baseclass

    Args:
        test_indices (np.ndarray, optional): Test for changes with these
            indices of Y. If not provided, all series in Y will be used as
            test indices
        estimator (sklearn compatible estimator): estimation model from
            scikit-learn used to fit and predict timeseries
            (default: ``Lasso(alpha=20)``)

    Attributes:
        record_template (np.ndarray): An empty NumPy structured array that is
            a template for the model's ``record``
        models (np.ndarray): prediction model objects
        record (np.ndarray): NumPy structured array containing timeseries model
            attribute information
        n_series (int): number of bands in Y
        n_features (int): number of coefficients in X design matrix

    Methods:
        setup(self, df, **config): Configure model and (optionally) return
            design matrix (or None if not required for model)
        preprocess(self, X, Y, dates, **config): Preprocess a unit area of
            data (pixel, segment, etc.) (e.g., mask, scale, transform, etc.)
        fit(self, X, Y, dates): Fit timeseries model
        fit_models(self, X, Y, bands=None): Fit timeseries
            models for ``bands`` within ``Y`` for a given ``X``
        predict(self, X, dates, series=None): Return a 2D NumPy
            array of y-hat predictions of requested series for a given X
        score(self, X, Y, dates): Return timeseries model
            performance scores
        plot(self, X, Y, dates, **config): Plot the timeseries model
            results

    Record structured arrays must contain the following:

        * start (int): starting dates of timeseries segments
        * end (int): ending dates of timeseries segments
        * break (int): break dates of timeseries segments
        * coef (n x p double): number of bands x number of features
          coefficients matrix for predictions
        * rmse (n double): Root Mean Squared Error for each band
        * px (int): pixel X coordinate
        * py (int): pixel Y coordinate

    """

    px = 0
    py = 0
    n_series = 0
    n_features = 0

    def __init__(self,
                 test_indices=None,
                 estimator=sklearn.linear_model.Lasso(alpha=20),
                 **kwargs):
        self.test_indices = np.asarray(test_indices)
        self.estimator = sklearn.clone(estimator)
        self.models = []  # leave empty, fill in during `fit`

        self.n_record = 0
        self.record = []

    @property
    def record_template(self):
        """ Return a YATSM record template for features in X and series in Y

        Record template will set `px` and `py` if defined as class attributes.
        Otherwise `px` and `py` coordinates will default to 0.

        Returns:
            np.ndarray: NumPy structured array containing a template of a YATSM
                record

        """
        record_template = np.zeros(1, dtype=[
            ('start', 'i4'),
            ('end', 'i4'),
            ('break', 'i4'),
            ('coef', 'float32', (self.n_coef, self.n_series)),
            ('rmse', 'float32', (self.n_series)),
            ('px', 'u2'),
            ('py', 'u2')
        ])
        record_template['px'] = getattr(self, 'px', 0)
        record_template['py'] = getattr(self, 'py', 0)

        return record_template

# SETUP & PREPROCESSING
    def setup(self, df, **config):
        """ Setup model for input dataset and (optionally) return design matrix

        Args:
            df (pandas.DataFrame): Pandas dataframe containing dataset
                attributes (e.g., dates, image ID, path/row, metadata, etc.)
            config (dict): YATSM configuration dictionary from user, including
                'dataset' and 'YATSM' sub-configurations

        Returns:
            X (np.ndarray, or None): return design matrix if used by algorithm

        """
        X = patsy.dmatrix(config['YATSM']['design_matrix'], data=df)
        return X

    def preprocess(self, X, Y, dates, **config):
        """ Preprocess a unit area of data (e.g., pixel, segment, etc.)

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (np.ndarray): ordinal dates for each observation in X/Y
            config (dict): YATSM configuration dictionary from user, including
                'dataset' and 'YATSM' sub-configurations

        """
        # Mask range of data
        valid = cyprep.get_valid_mask(
            Y,
            config['dataset']['min_values'],
            config['dataset']['max_values']).astype(bool)
        # Apply mask band
        idx_mask = config['dataset']['mask_band'] - 1
        valid *= np.in1d(Y.take(idx_mask, axis=0),
                         config['dataset']['mask_values'],
                         invert=True).astype(np.bool)

        Y = np.delete(Y, idx_mask, axis=0)[:, valid]
        X = X[valid, :]
        dates = dates[valid]

        return X, Y, dates

# TIMESERIES ENSEMBLE FIT/PREDICT
    def fit(self, X, Y, dates):
        """ Fit timeseries model

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (np.ndarray): ordinal dates for each observation in X/Y

        Returns:
            np.ndarray: NumPy structured array containing timeseries
                model attribute information

        """
        raise NotImplementedError('Subclasses should implement fit method')

    def fit_models(self, X, Y, bands=None):
        """ Fit timeseries models for `bands` within `Y` for a given `X`

        Updates or initializes fit for ``self.models``

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations) observation in the X design matrix
            bands (iterable): Subset of bands of `Y` to fit. If None are
                provided, fit all bands in Y

        """
        if bands is None:
            bands = np.arange(self.n_series)

        for b in bands:
            y = Y[b, :]

            model = self.models[b]
            model.fit(X, y)

            # Add in RMSE calculation
            model.rmse = rmse(y, model.predict(X))

            # Add intercept to intercept term of design matrix
            model.coef = model.coef_.copy()
            model.coef[0] += model.intercept_

    def predict(self, X, dates, series=None):
        """ Return a 2D NumPy array of y-hat predictions for a given X

        Predictions are made from ensemble of timeseries models such that
        predicted values are generated for each date using the model from the
        timeseries segment that intersects each date.

        Args:
            X (np.ndarray): Design matrix (number of observations x number of
                features)
            dates (int or np.ndarray): A single ordinal date or a np.ndarray of
                length X.shape[0] specifying the ordinal dates for each
                prediction
            series (iterable, optional): Return prediction for subset of series
                within timeseries model. If None is provided, returns
                predictions from all series

        Returns:
            Y (np.ndarray): Prediction for given X (number of series x number
                of observations)

        """
        raise NotImplementedError('Subclasses should implement "predict" '
                                  'method')

# DIAGNOSTICS
    def score(self, X, Y, dates):
        """ Return timeseries model performance scores

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (np.ndarray): ordinal dates for each observation in X/Y

        Returns:
            namedtuple: performance summary statistics

        """
        raise NotImplementedError('Subclasses should implement "score" method')

    def plot(self, X, Y, dates, **config):
        """ Plot the timeseries model results

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (np.ndarray): ordinal dates for each observation in X/Y
            config (dict): YATSM configuration dictionary from user, including
                'dataset' and 'YATSM' sub-configurations

        """
        raise NotImplementedError('Subclasses should implement "plot" method')

# MAKE ITERABLE
    def __iter__(self):
        """ Iterate over the timeseries segment records
        """
        for record in self.record:
            yield record

    def __len__(self):
        """ Return the number of segments in this timeseries model
        """
        return len(self.record)
