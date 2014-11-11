""" Helper classes for initializing a Scikit-Learn classifiers from an INI file
"""
from sklearn.ensemble import RandomForestClassifier

from ..utils import is_integer


class RandomForestHelper(RandomForestClassifier):
    """A random forest classifier wrapper

    This helper class wraps the RandomForestClassifier from `scitkit-learn`
    so that it can initialize from an INI configuration file.

    For documentation on RandomForestClassifier, see:

    `sklearn.ensemble.RandomForestClassifier
    <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_

    Args:
      config (configparser.ConfigParser): Configuration parser to retrieve
        parameters from

    """

    # Setup defaults for ConfigParser to use
    defaults = {
        'n_estimators': 500,
        'criterion': 'gini',
        'max_features': 'auto',
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_leaf_nodes': None,
        'bootstrap': True,
        'oob_score': True,
        'n_jobs': 1
    }

    def __init__(self, config):
        super(RandomForestHelper, self).__init__(
            n_estimators=config.getint('init', 'n_estimators'),
            bootstrap=config.getboolean('init', 'bootstrap'),
            oob_score=config.getboolean('init', 'oob_score'),
            n_jobs=config.getint('init', 'n_jobs')
        )

        self.criterion = config.get('init', 'criterion')
        self.max_depth = (None if config.get('init', 'max_depth') == 'None'
                          else config.getint('init', 'max_depth'))
        self.min_samples_split = config.getint('init', 'min_samples_split')
        self.min_samples_leaf = config.getint('init', 'min_samples_leaf')
        max_features = config.get('init', 'max_features')
        if isinstance(max_features, str):
            if max_features == 'None':
                self.max_features = None
            elif max_features in ['auto', 'sqrt', 'log2']:
                self.max_features = max_features
            else:
                raise ValueError(
                    'Invalid value for max_features. Allowed string '
                    'values are "auto", "sqrt" or "log2".')
        elif is_integer(max_features):
            self.max_features = int(max_features)
        else:
            # must be float
            self.max_features = float(max_features)

        self.max_leaf_nodes = (
            None if config.get('init', 'max_leaf_nodes') == 'None'
            else config.getint('init', 'max_leaf_nodes')
        )

        # Check for options in fit method
        # TODO
#        if config.has_option('fit', 'sample_weight'):
#
