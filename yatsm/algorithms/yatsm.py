""" Yet Another TimeSeries Model baseclass
"""
import numpy as np
import sklearn
import sklearn.linear_model


class YATSM(object):
    """ Yet Another TimeSeries Model baseclass

    Args:
      fit_indices (np.ndarray): Fit models for these indices of Y
      design_info (patsy.DesignInfo): Patsy design information for X
      test_indices (np.ndarray, optional): Test for changes with these indices
        of Y. If not provided, all `fit_indices` will be used as test indices
      lm (sklearn.linear_model predictor): regression model from scikit-learn
        used to fit and predict timeseries (default: `Lasso(alpha=20)`)

    Attributes:
      models (list): `sklearn` model objects used for fitting / prediction
      record (np.ndarray): NumPy structured array containing timeseries
        model attribute information

    Record structured arrays must contain the following:

        * start (int): starting dates of timeseries segments
        * end (int): ending dates of timeseries segments
        * break (int): break dates of timeseries segments
        * coef (nb x p double): number of bands x number of features
            coefficients matrix for predictions
        * px (int): pixel X coordinate
        * py (int): pixel Y coordinate

    """

    def __init__(self, fit_indices, design_info, test_indices=None,
                 lm=sklearn.linear_model.Lasso(alpha=20),
                 **kwargs):
        self.fit_indices = np.asarray(fit_indices)

        self.design_info = design_info
        if 'x' not in design_info.term_name_slices.keys():
            raise AttributeError('Design info must specify "x" (slope)')
        self.i_x = design_info.term_name_slices['x'].start

        self.test_indices = np.asarray(test_indices)
        if self.test_indices is None:
            self.test_indices = self.fit_indices

        self.lm = sklearn.clone(lm)

        self.record_template = np.zeros(1, dtype=[
            ('start', 'i4'),
            ('end', 'i4'),
            ('break', 'i4'),
            ('coef', 'float32', (len(self.design_info.column_names),
                                 len(self.fit_indices))),
            ('px', 'u2'),
            ('py', 'u2')
        ])
        self.record_template['px'] = kwargs.get('px', 0)
        self.record_template['py'] = kwargs.get('py', 0)
        self.n_record = 0
        self.record = np.copy(self.record_template)

# TIMESERIES ENSEMBLE FIT/PREDICT
    def fit(self, X, Y):
        """ Fit timeseries model

        Args:
          X (np.ndarray): design matrix (number of observations x number of
            features)
          Y (np.ndarray): independent variable matrix (number of series x number
            of observations)

        """
        raise NotImplementedError('Subclasses should implement fit method')

    def predict(self, X, series=None, dates=1):
        """ Return a 2D NumPy array of y-hat predictions for a given X

        Predictions are made from ensemble of timeseries models such that
        predicted values are generated for each date using the model from the
        timeseries segment that intersects each date.

        Args:
          X (np.ndarray): Design matrix (number of observations x number of
            features)
          series (iterable, optional): Return prediction for subset of series
            within timeseries model. If None is provided, returns predictions
            from all series
          dates (int or np.ndarray, optional): Index of `X`'s features or a
            np.ndarray of length X.shape[0] specifying the ordinal dates for
            each

        Returns:
          Y (np.ndarray): Prediction for given X (number of series x number of
            observations)

        """
        raise NotImplementedError('Have not implemented this function yet')

    def fit_models(self, X, Y, bands=None):
        """ Fit timeseries models for `bands` within `Y` for a given `X`

        Args:
            X (np.ndarray): design matrix (number of observations x number of
                features)
            Y (np.ndarray): independent variable matrix (number of series x
                number of observations) observation in the X design matrix
            bands (iterable): Subset of bands of `Y` to fit. If None are
                provided, fit all bands in Y

        Returns:
            np.ndarray: fitted model objects

        """
        if bands is None:
            bands = self.fit_indices

        models = []
        for b in bands:
            y = Y.take(b, axis=0)
            model = sklearn.clone(self.lm).fit(X, y)

            # Add in RMSE calculation  # TODO: numba?
            model.rmse = ((y - model.predict(X)) ** 2).mean(axis=0) ** 0.5

            # Add intercept to intercept term of design matrix
            model.coef = model.coef_.copy()
            model.coef[0] += model.intercept_

            models.append(model)

        return np.array(models)

# DIAGNOSTICS
    def score(self, X, Y):
        """ Returns some undecided description of timeseries model performance

        Args:
          X (np.ndarray): design matrix (number of observations x number of
            features)
          Y (np.ndarray): independent variable matrix (number of series x number
            of observations)

        Returns:
          float: some sort of performance summary statistic

        """
        raise NotImplementedError('Have not implemented this function yet')

    def plot(self, X, Y):
        """ Plot the timeseries model results

        Args:
          X (np.ndarray): design matrix (number of observations x number of
            features)
          Y (np.ndarray): independent variable matrix (number of series x number
            of observations)

        """
        raise NotImplementedError('Have not implemented this function yet')

# MAKE ITERABLE
    def __iter__(self):
        """ Iterate over the timeseries segment records
        """
        for record in self.record:
            yield record

    def __len__(self):
        """ Return the number of segments in this timeseries model
        """
        return self.record.shape[0]
