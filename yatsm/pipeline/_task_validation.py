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

import decorator
import future.utils

from .language import OUTPUT, REQUIRE, DATA, RECORD
from ..errors import PipelineConfigurationError as PCError


REQUIRED_BY_DEFAULT = True


def eager_task(func):
    """ A task decorator that declares it can compute all pixels at once
    """
    func.is_eager = True
    return func


def _parse_signature(signature, req_len=None):
    """ Parse a signature for basic validity and structure

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
    # Given as <str:name>=[<object>, ...]
    if isinstance(signature, list):
        signature = (REQUIRED_BY_DEFAULT, signature)

    # Given as <str:name>=(<bool:required>, [<object>, ...]])
    if (isinstance(signature, tuple) and len(signature) == 2 and
          isinstance(signature[0], bool) and isinstance(signature[1], list)):
        if req_len and len(signature[1]) != req_len:
            raise PCError("Function signature must have {n} arguments"
                          .format(n=req_len))
        return signature
    else:
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
            # If the specification description has a specific length requirement
            if isinstance(desc, Iterable):
                if desc and len(value) != len(desc):
                    raise ValueError(
                        "Specification requires {n} elements ({desc}) but "
                        "{n2} elements were passed"
                        .format(n=len(desc), desc=desc, n2=len(value)))


def check(name, **signature):
    """ Validate inputs to argument `name`

    Args:
        name (str): Name of argument (this argument should expect a `dict`)
        signature (dict): Keyword arguments gathered

    Raises:
        PipelineConfigurationError: Raise if function use doesn't match
            required signature. Note that this error is a subclass of TypeError

    """
    # Allow:
    #   1) Explicit "required": `check('x', data=(False, [str]))`
    #   2) Assume default (not required): `check('x', data=[str])`
    for k, sig in signature.items():
        if k not in (DATA, RECORD):
            raise PCError(
                'Unknown signature check for argument named "{name}"'
                .format(name=k))
        req_len = 1 if name == OUTPUT and k == RECORD else None
        try:
            signature[k] = _parse_signature(sig, req_len=req_len)
        except PCError as pce:
            future.utils.raise_with_traceback(PCError(
                "'{}' signature, '{}', is invalid: {}".format(k, sig, pce)
            ))

    @decorator.decorator
    def wrapper(func, *args, **kwargs):
        arg_names, va_args, va_kwargs, _ = inspect.getargspec(func)
        if name not in arg_names:
            raise PCError("Argument specified, '{}', does not match "
                          "function call signature".format(name))
        arg_idx = arg_names.index(name)
        arg = args[arg_idx]

        try:
            _validate_specification(arg, signature)
        except Exception as exc:
            future.utils.raise_with_traceback(PCError(
                "Argument '{}' to '{}' is invalid:: {}: {}"
                .format(name, func.__name__, exc.__class__.__name__, exc)))
        return func(*args, **kwargs)

    return wrapper


#: Decorator to check inputs to `output` argument
outputs = functools.partial(check, OUTPUT)
#: Decorator to check inputs to `require` argument
requires = functools.partial(check, REQUIRE)
