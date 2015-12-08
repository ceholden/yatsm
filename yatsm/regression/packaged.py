""" Module for finding regression algorithms packaged with YATSM
"""
import json
import logging
import os
import pkg_resources

logger = logging.getLogger('yatsm')

# packaged_regressions = ['OLS', 'sklearn_Lasso20', 'glmnet_Lasso20',
#                         'glmnet_LassoCV_n50', 'glmnet_LassoCV_n100',
#                         'rlm_maxiter10']
packaged_regressions = []
_packaged = pkg_resources.resource_filename(
    __package__, os.path.join('pickles', 'pickles.json'))
if pkg_resources.resource_exists(__package__,
                                 os.path.join('pickles', 'pickles.json')):
    with open(_packaged, 'r') as f:
        packaged_regressions = json.load(f).keys()


def find_packaged_regressor(name):
    """ Find location of a regression method packaged with YATSM

    See ``yatsm.regression.packaged.packaged_regressions`` for a list of
    available pre-packaged regressors

    Args:
        name (str): name of packaged regression object

    Returns:
        str: path to packaged regression method

    Raises:
        KeyError: raise KeyError if user specifies unknown regressor
        IOError: raise IOError if the packaged regressor cannot be found

    """
    if name not in packaged_regressions:
        raise KeyError('Cannot load unknown packaged regressor %s' % name)

    path = pkg_resources.resource_filename(__name__, 'pickles')
    logger.debug('Checking data files in %s for packaged regressors' % path)
    if not pkg_resources.resource_exists(__name__, 'pickles'):
        raise IOError('Cannot find packaged regressors in %s. Did you install '
                      'YATSM via setuptools?' % path)

    resource = os.path.join('pickles', name + '.pkl')
    if not pkg_resources.resource_exists(__name__, resource):
        raise IOError('Cannot find packaged regression method %s, but package '
                      'directory exists. Check the contents of %s if possible'
                      % (resource, path))

    return pkg_resources.resource_filename(__name__, resource)
