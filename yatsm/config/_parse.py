""" Configuration file validation, including additional specific steps
"""
import jsonschema
import pkgutil
import yaml
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from ._util import expand_envvars, validate_with_defaults
from ..errors import InvalidConfigurationException


PARSERS = []


def _validate_config(config):
    try:
        schema = yaml.load(
            pkgutil.get_data('yatsm', 'config/config_schema.yaml'),
            Loader=SafeLoader
        )
        config = validate_with_defaults(config, schema)
    except jsonschema.ValidationError as exc:
        raise InvalidConfigurationException(exc)
    return config


def validate_and_parse_configfile(path, parsers=None):
    """ Validate and parse configuration file

    Args:
        path (str): Filename
        parsers (list): A list of functions to apply over a configuration
            (``dict``) before returning
    Returns:
        dict: Configuration data
    Raises:
        InvalidConfigurationException: Raise if configuration file does not
            validate against an expected schema
    """
    if not parsers:
        parsers = PARSERS

    with open(path, 'rb') as fid:
        config = yaml.load(fid, Loader=SafeLoader)

    config = expand_envvars(config)
    config = _validate_config(config)

    for parse in parsers:
        config = parse(config)

    return config
