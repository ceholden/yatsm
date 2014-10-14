""" Random Forest algorithm helper class """
from sklearn.ensemble import RandomForestClassifier


def is_integer(s):
    """ Returns True if `s` is an integer """
    try:
        int(s)
        return True
    except:
        return False


class RandomForestHelper(RandomForestClassifier):

    # Preserve __doc__
    __doc__ += RandomForestClassifier.__doc__
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
        """ Override __init__ to work with config file """
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
