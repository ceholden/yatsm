""" Yet Another TimeSeries Model baseclass
"""
import numpy as np
import patsy
import sklearn
import sklearn.linear_model

from .._cyprep import get_valid_mask
from ..regression.diagnostics import rmse
from ..regression.transforms import harm  # noqa


class YATSM(object):
    """ Yet Another TimeSeries Model baseclass

    .. note::
        When ``YATSM`` objects are fit, the intended order of method calls is:

            1. Setup the model with :func:`~setup`
            2. Preprocess a time series for one unit area with
               :func:`~preprocess`
            3. Fit the time series with the YATSM model using :func:`~fit`
            4. A fitted model can be used to

                * Predict on additional design matrixes with :func:`~predict`
                * Plot diagnostic information with :func:`~plot`
                * Return goodness of fit diagnostic metrics with :func:`~score`

    .. note::
        Record structured arrays must contain the following:

            * ``start`` (`int`): starting dates of timeseries segments
            * ``end`` (`int`): ending dates of timeseries segments
            * ``break`` (`int`): break dates of timeseries segments
            * ``coef`` (`double (n x p shape)`): number of bands x number of
              features coefficients matrix for predictions
            * ``rmse`` (`double (n length)`): Root Mean Squared Error for each
              band
            * ``px`` (`int`): pixel X coordinate
            * ``py`` (`int`): pixel Y coordinate

    Args:
        test_indices (numpy.ndarray): Test for changes with these
            indices of ``Y``. If not provided, all series in ``Y`` will be used
            as test indices
        estimator (dict): dictionary containing estimation model from
            ``scikit-learn`` used to fit and predict timeseries and,
            optionally, a dict of options for the estimation model ``fit``
            method (default: ``{'object': Lasso(alpha=20), 'fit': {}}``)
        kwargs (dict): dictionary of addition keyword arguments
            (for sub-classes)

    Attributes:
        record_template (numpy.ndarray): An empty NumPy structured array that
            is a template for the model's ``record``
        models (numpy.ndarray): prediction model objects
        record (numpy.ndarray): NumPy structured array containing timeseries
            model attribute information
        n_record (int): number of recorded segments in time series model
        n_series (int): number of bands in ``Y``
        px (int): pixel X location or index
        n_features (int): number of coefficients in ``X`` design matrix
        py (int): pixel Y location or index

    """

    def __init__(self,
                 test_indices=None,
                 estimator={'object': sklearn.linear_model.Lasso(alpha=20),
                            'fit': {}},
                 **kwargs):
        self.test_indices = np.asarray(test_indices)
        self.estimator = sklearn.clone(estimator['object'])
        self.estimator_fit = estimator.get('fit', {})
        self.models = []  # leave empty, fill in during `fit`

        self.n_record = 0
        self.record = []

        self.n_series, self.n_features = 0, 0
        self.px = kwargs.get('px', 0)
        self.py = kwargs.get('py', 0)

    @property
    def record_template(self):
        """ YATSM record template for features in X and series in Y

        Record template will set ``px`` and ``py`` if defined as class
        attributes. Otherwise ``px`` and ``py`` coordinates will default to 0.

        Returns:
            numpy.ndarray: NumPy structured array containing a template of a
                YATSM record

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
        record_template['px'] = self.px
        record_template['py'] = self.py

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
            numpy.ndarray or None: return design matrix if used by algorithm
        """
        X = patsy.dmatrix(config['YATSM']['design_matrix'], data=df)
        return X

    def preprocess(self, X, Y, dates,
                   min_values=None, max_values=None,
                   mask_band=None, mask_values=None, **kwargs):
        """ Preprocess a unit area of data (e.g., pixel, segment, etc.)

        This preprocessing step will remove all observations that either
        fall outside of the minimum/maximum range of the data or are flagged
        for masking in the ``mask_band`` variable in ``Y``. If ``min_values``
        or ``max_values`` are not specified, this masking step is skipped.
        Similarly, masking based on a QA/QC or cloud mask will not be performed
        if ``mask_band`` or ``mask_values`` are not provided.

        Args:
            X (numpy.ndarray): design matrix (number of observations x number
                of features)
            Y (numpy.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (numpy.ndarray): ordinal dates for each observation in X/Y
            min_values (np.ndarray): Minimum possible range of values for each
                variable in Y (optional)
            max_values (np.ndarray): Maximum possible range of values for each
                variable in Y (optional)
            mask_band (int): The mask band in Y (optional)
            mask_values (sequence): A list or np.ndarray of values in the
                ``mask_band`` to mask (optional)

        Returns:
            tuple (np.ndarray, np.ndarray, np.ndarray): X, Y, and dates after
                being preprocessed and masked

        """
        if min_values is None or max_values is None:
            valid = np.ones(dates.shape[0], dtype=np.bool)
        else:
            # Mask range of data
            valid = get_valid_mask(Y, min_values, max_values).astype(bool)

        # Apply mask band
        if mask_band is not None and mask_values is not None:
            idx_mask = mask_band - 1
            valid *= np.in1d(Y.take(idx_mask, axis=0), mask_values,
                             invert=True).astype(np.bool)

        Y = np.delete(Y, idx_mask, axis=0)[:, valid]
        X = X[valid, :]
        dates = dates[valid]

        return X, Y, dates

# TIMESERIES ENSEMBLE FIT/PREDICT
    def fit(self, X, Y, dates):
        """ Fit timeseries model

        Args:
            X (numpy.ndarray): design matrix (number of observations x number
                of features)
            Y (numpy.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (numpy.ndarray): ordinal dates for each observation in X/Y

        Returns:
            numpy.ndarray: NumPy structured array containing timeseries
                model attribute information

        """
        raise NotImplementedError('Subclasses should implement fit method')

    def fit_models(self, X, Y, bands=None):
        """ Fit timeseries models for `bands` within `Y` for a given `X`

        Updates or initializes fit for ``self.models``

        Args:
            X (numpy.ndarray): design matrix (number of observations x number
                of features)
            Y (numpy.ndarray): independent variable matrix (number of series x
                number of observations) observation in the X design matrix
            bands (iterable): Subset of bands of `Y` to fit. If None are
                provided, fit all bands in Y

        """
        if bands is None:
            bands = np.arange(self.n_series)

        for b in bands:
            y = Y[b, :]

            model = self.models[b]
            model.fit(X, y, **self.estimator_fit)

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
            X (numpy.ndarray): Design matrix (number of observations x number
                of features)
            dates (int or numpy.ndarray): A single ordinal date or a np.ndarray
                of length X.shape[0] specifying the ordinal dates for each
                prediction
            series (iterable, optional): Return prediction for subset of series
                within timeseries model. If None is provided, returns
                predictions from all series

        Returns:
            numpy.ndarray: Prediction for given X (number of series x number of
                observations)

        """
        raise NotImplementedError('Subclasses should implement "predict" '
                                  'method')

# DIAGNOSTICS
    def score(self, X, Y, dates):
        """ Return timeseries model performance scores

        Args:
            X (numpy.ndarray): design matrix (number of observations x number
                of features)
            Y (numpy.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (numpy.ndarray): ordinal dates for each observation in X/Y

        Returns:
            namedtuple: performance summary statistics

        """
        raise NotImplementedError('Subclasses should implement "score" method')

    def plot(self, X, Y, dates, **config):
        """ Plot the timeseries model results

        Args:
            X (numpy.ndarray): design matrix (number of observations x number
                of features)
            Y (numpy.ndarray): independent variable matrix (number of series x
                number of observations)
            dates (numpy.ndarray): ordinal dates for each observation in X/Y
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
