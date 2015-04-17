from glmnet_fit import GLMLasso
from robust_fit import RLM, bisquare


# Possible coefficients
design_coefs = ['all',
                'intercept', 'slope',
                'seasonality', 'categorical',
                'rmse']


def _key_lookup_ignorecase(d, key):
    """ Search dict for key, ignoring case

    Args:
      d (dict): dict to search
      key (str): key to search for

    Returns:
      str or None: key in dict `d` matching `key` if found; else None

    """
    key = [k for k in d.keys() if key.lower() == k.lower()]
    if key:
        return key[0]
    else:
        return None


def design_to_indices(design_matrix, features):
    """ Return indices of coefficients for features in design matrix

    Args:
      design_matrix (OrderedDict): OrderedDict containing design features keys
        and indices of coefficient matrix as values
      features (list): list of feature coefficients to extract

    Return:
      tuple: list of indices and names for each feature specified in `features`

    """
    if 'all' in features:
        features = design_coefs[1:]

    i_coefs = []
    coef_names = []
    for c in features:
        if c == 'intercept':
            k = _key_lookup_ignorecase(design_matrix, 'intercept')
            i_coefs.append(design_matrix.get(k))
            coef_names.append(k)
        elif c == 'slope':
            k = _key_lookup_ignorecase(design_matrix, 'x')
            i_coefs.append(design_matrix.get(
                _key_lookup_ignorecase(design_matrix, 'x')))
            coef_names.append(k)
        elif c == 'seasonality':
            i = [k for k in design_matrix.keys() if 'harm' in k]
            i_coefs.extend([design_matrix[_i] for _i in i])
            coef_names.extend(i)
        elif c == 'categorical':
            i = [k for k in design_matrix.keys() if 'C' in k]
            i_coefs.extend([design_matrix[_i] for _i in i])
            coef_names.extend(i)

    i_coefs = [i for i in i_coefs if i is not None]
    coef_names = [n for n in coef_names if n is not None]

    return i_coefs, coef_names
