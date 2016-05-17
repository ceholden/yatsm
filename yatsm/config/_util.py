""" Utilities for configuration file validation via jsonschema
"""
from functools import partial
import logging
import os

from jsonschema import Draft4Validator, validators, validate
import six

logger = logging.getLogger(__name__)


def extend_with_default(validator_cls):
    """ Applies default field from jsonschema

    Copied, with modificatios, from python-jsonschema FAQ. See:
    https://python-jsonschema.readthedocs.io/en/latest/faq/#why-doesn-t-my-schema-that-has-a-default-property-actually-set-the-default-on-my-instance
    """
    validate_props = validator_cls.VALIDATORS['properties']
    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in six.iteritems(properties):
            if 'default' in subschema:
                instance.setdefault(prop, subschema['default'])

        for error in validate_props(validator, properties, instance, schema):
            yield error

    return validators.extend(validator_cls, {'properties': set_defaults})


DefaultValidatingDraft4Validator = extend_with_default(Draft4Validator)

#: jsonschema.validate called with instance of DefaultValidatingDraft4Validator
def validate_with_defaults(schema, config):
    return DefaultValidatingDraft4Validator(schema).validate(config)
#validate_with_defaults = partial(validate,
#                                cls=DefaultValidatingDraft4Validator)


def expand_envvars(d):
    """ Recursively convert lookup that look like environment vars in a dict

    This function things that environmental variables are values that begin
    with `$` and are evaluated with :func:`os.path.expandvars`. No exception
    will be raised if an environment variable is not set.

    Args:
        d (dict): expand environment variables used in the values of this
            dictionary

    Returns:
        dict: input dictionary with environment variables expanded

    """
    def check_envvar(k, v):
        """ Warn if value looks un-expanded """
        if '$' in v:
            logger.warning('Config key=value pair might still contain '
                           'environment variables: "%s=%s"' % (k, v))

    _d = d.copy()
    for k, v in six.iteritems(_d):
        if isinstance(v, dict):
            _d[k] = expand_envvars(v)
        elif isinstance(v, str):
            _d[k] = os.path.expandvars(v)
            check_envvar(k, _d[k])
        elif isinstance(v, (list, tuple)):
            n_v = []
            for _v in v:
                if isinstance(_v, str):
                    _v = os.path.expandvars(_v)
                    check_envvar(k, _v)
                n_v.append(_v)
            _d[k] = n_v
    return _d
