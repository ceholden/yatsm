""" Configuration file validation, including additional specific steps
"""
import jsonschema
import pkgutil
import yaml
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from yatsm.config._util import expand_envvars, validate_with_defaults
from yatsm.errors import InvalidConfigurationException


PARSERS = [expand_envvars]


def load_config(path):
    """ Safely load a YATSM configuration file

    Args:
        path (str): Filename

    Returns:
        dict: Configuration data

    Raises:
        InvalidConfigurationException: Raise if filename provided
        cannot be parsed into a `dict` by :ref:`yaml.load`

    """
    with open(str(path), 'rb') as fid:
        config = yaml.load(fid, Loader=SafeLoader)

    # yaml.load won't complain if it gets garbage, but return type
    # won't be a dict
    if not isinstance(config, dict):
        raise InvalidConfigurationException('Could not parse configuration '
                                            'file "{0}" to a `dict`'
                                            .format(path))
    return config


def validate_config(data):
    """ Validate and parse configuration file

    Checks against schema defined in `yatsm/config/config_schema.yaml`
    and will fill in default values missing from user input.

    Args:
        data (dict): A configuration `dict` to validate

    Returns:
        dict: Configuration data

    Raises:
        InvalidConfigurationException: Raise if configuration file does not
        validate against an expected schema
    """
    try:
        schema_data = pkgutil.get_data('yatsm', 'config/config_schema.yaml')
        schema = yaml.load(schema_data, Loader=SafeLoader)
        return validate_with_defaults(data, schema)
    except jsonschema.ValidationError as exc:
        raise InvalidConfigurationException(exc.message)


def parse_config(data, parsers=None):
    """ Read and parse a configuration file

    By default, parsing includes filling in any environment variables
    used in :ref:`data`

    Args:
        data (dict): Configuration data in a `dict`
        parsers (list): A list of functions to apply over :ref:`data`
            before returning. Defaults to :ref:`PARSERS`.

    Returns:
        dict: Configuration data
    """
    if not parsers:
        parsers = PARSERS

    data = expand_envvars(data)

    for parse in parsers:
        data = parse(data)

    return data
