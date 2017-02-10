""" Decorators for validating task requirements and outputs

This validation should cause errors to occur when gathering an assortment of
delayed tasks, and not when executed.

Validation of requirements and outputs may be specified in terms of the
`data` and `record` information that they interact with. A normalized
difference vegetation index, for example, requires two spectral bands and
outputs a single index. We want to allow the user to specify the names of the
input spectral bands and the name of the output transformation to make this
function more generalized and usable in a pipeline. The fact that this wrapper
function requires two input strings (as the spectral band names in an
``xarray.Dataset``, for example) can be specified using the ``@require``
decorator function, an indicator for either `data` or `record` structures,
and a ``list`` of objects.

A task may only define one `record` output, so the output type for this should
just be ``[str]``.

The validation for this task may be specified as::

.. code-block:: python

    @requires(data=[str, str])
    @outputs(data=[str])
    def norm_diff(work, require, output, **config):
        ...

The objects used here were of type ``type``, but one might use ``str`` objects
if the names were suggestive of a particular intended meaning::

.. code-block:: python

    @requires(data=['nir', 'red'])
    @outputs(data=['ndvi'])
    def ndvi(work, require, output, **config):
        ...

Some tasks may allow a variable number of inputs. This may be accomplished
by providing an empty list, as demonstrated below for the `data` requirement::

.. code-block:: python

    @requires(data=[])
    @outputs(data=[str])
    def sum_all_spectral_bands(work, require, output, **config):
        ...

If a task allows a requirement or output, but this requirement or output is
optional, then it may be specified as optional using the full syntax in a tuple
``(bool: required, signature)``. By default, we assume that requirements and
output specifications are required.

.. code-block:: python

    @requires(data=[str, str],
             record=(False, [str]))
    def some_task(work, require, output, **config):
        ...

"""
from collections import Iterable
import functools
import inspect
import logging

import decorator
import six

from yatsm.pipeline.language import PIPE_CONTENTS, REQUIRE, OUTPUT, STASH
from yatsm.errors import PipelineConfigurationError as PCError

logger = logging.getLogger(__name__)

REQUIRED_BY_DEFAULT = True


def eager_task(func):
    """ A task decorator that declares it can compute all pixels at once
    """
    func.is_eager = True
    return func


def version(version_str):
    def decorator(func):
        func.version = version_str
        return func
    return decorator


def _parse_signature(signature, req_len=None):
    """ Parse a signature for basic validity and structure

    Example:

    .. code-block:: python

        >>> (True, [str, str])  # two required arguments
        >>> [str, str]  # two arguments, required by default
        >>> (False, [str, str])  # two optional arguments

    Args:
        signature (iterable): One or more objects in a ``list``, or a
            ``tuple`` of (``bool``, ``list``) describing if the signature
            is required
        req_len (int): Requirement for the length of the description in
            ``signature``

    Returns:
        tuple[bool, list]: The signature

    Raises:
        KeyError: If ``name`` isn't a supported type of function signature
        TypeError: If ``signature`` is invalid
    """
    def _has_required(l):
        return l and isinstance(l[0], bool)

    def _check(l):
        return (
            isinstance(l, (tuple, list, )) and
            len(l) == 2 and
            isinstance(l[0], bool) and
            isinstance(l[1], (tuple, list, ))
        )

    if isinstance(signature, (tuple, list, )):
        # Given as <str:name>=[<object>, ...]
        if not _has_required(signature):
            signature = (REQUIRED_BY_DEFAULT, signature)
        # Given as <str:name>=(<bool:required>, [<object>, ...]])
        try:
            okay = _check(signature)
        except Exception as exc:
            logger.exception('Invalid signature {0}'.format(signature), exc)
            raise
        else:
            if okay:  # and fall if not
                return signature
    raise PCError("Invalid signature: {sig}".format(sig=signature))


def _validate_specification(spec, signature):
    if not isinstance(spec, dict):
        raise TypeError(" should be a dictionary")

    for name, (required, desc) in signature.items():
        if required and name not in spec:
            raise KeyError("Required attribute, '{}', not passed to function"
                           .format(name))
        elif name in spec:
            value = spec[name]
            # If the specification description has a specific length
            # requirement
            if isinstance(desc, Iterable):
                if desc and len(value) != len(desc):
                    raise ValueError(
                        "Specification requires {n} elements ({desc}) but "
                        "{n2} elements were passed"
                        .format(n=len(desc), desc=desc, n2=len(value)))


def check(name, **signature):
    """ Validate inputs to argument `name`

    Example:

    .. code-block:: python

        # required, by default, see `REQUIRED_BY_DEFAULT`
        @check('var', data=[str, str])
        # not required, explicitly
        @check('var', data=(False, [str, str]))

    Args:
        name (str): Name of argument (this argument should expect a `dict`)
        signature (dict): Keyword arguments gathered

    Raises:
        PipelineConfigurationError: Raise if function use doesn't match
            required signature. Note that this error is a subclass of TypeError
    """
    # Allow:
    #   1) Explicit "required": `check('x', data=(False, [str]))`
    #   2) Assume default (required): `check('x', data=[str])`
    for key, sig in signature.items():
        if key not in PIPE_CONTENTS:
            raise PCError('Unknown argument "{0}" to check'.format(key))
        try:
            signature[key] = _parse_signature(sig)
        except Exception as exc:
            logger.exception('Invalid signature: {0}'.format(sig), exc)
            six.raise_from(PCError('Invalid signature: {0}'.format(sig)), exc)

    @decorator.decorator
    def wrapper(func, *args, **kwargs):
        arg_names, va_args, va_kwargs, _ = inspect.getargspec(func)
        if name not in arg_names:
            raise PCError("Arg specified, '{0}', does not match "
                          "function call signature".format(name))
        arg_idx = arg_names.index(name)
        arg = args[arg_idx]

        try:
            _validate_specification(arg, signature)
        except Exception as exc:
            logger.exception('Arg "{0}" to "{1.__name__}" is invalid'
                             .format(name, func), exc)
            six.raise_from(PCError('Arg "{0}" to "{1.__name__}" is invalid'
                                   .format(name, func)), exc)
        return func(*args, **kwargs)

    return wrapper


#: Decorator to check inputs to `output` argument
outputs = functools.partial(check, OUTPUT)
#: Decorator to check inputs to `require` argument
requires = functools.partial(check, REQUIRE)
#: Decorator to check inputs to `stash` argument
stash = functools.partial(check, STASH)
