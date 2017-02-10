""" Tests for yatsm.pipeline._task_validation
"""
import pytest
import toolz

from yatsm.pipeline._task_validation import (check, requires, outputs,
                                             REQUIRE, OUTPUT)
from yatsm.errors import PipelineConfigurationError


def func_sig(work, require, output, **config):
    return work


def test_check_pass_1():
    # Fully specified
    f = check(REQUIRE, data=(True, [str, str]))(func_sig)
    f({}, {'data': ['red', 'nir']}, {})


def test_check_pass_2():
    # Partially specified -- assume 'data' isn't required
    f = check(REQUIRE, data=(False, [str, str]))(func_sig)
    f({}, {'data': ['red', 'nir']}, {})


def test_check_pass_3():
    # Two checks -- a require and output
    f = check(REQUIRE,
              data=(True, [str, str]),
              record=(True, [str]))(func_sig)
    f({}, {'data': ['red', 'nir'], 'record': ['ccdc']}, {})


def test_check_fail_1():
    # Invalid requirement signature
    with pytest.raises(PipelineConfigurationError) as exc:
        check(REQUIRE, data='asdf')(func_sig)
    assert 'invalid signature' in str(exc).lower()


def test_check_fail_2():
    # Invalid 'require' argument
    f = check(REQUIRE, data=(True, ['red', 'nir']))(func_sig)
    with pytest.raises(PipelineConfigurationError) as exc:
        f({}, 'not a dict', {})
    assert 'should be a dictionary' in str(exc)


def test_check_fail_3():
    # Missing required attribute (arg requirement-ness defaults to True)
    f = check(REQUIRE, data=['red', 'nir'])(func_sig)
    with pytest.raises(PipelineConfigurationError) as exc:
        f({}, {'record': []}, {})
    assert 'KeyError' in str(exc)
    assert 'not passed to function' in str(exc)


def test_check_fail_4():
    # Wrong number of arguments
    f = check(REQUIRE, data=[str, str])(func_sig)
    with pytest.raises(PipelineConfigurationError) as exc:
        f({}, {'data': ['asdf', 'red', 'nir']}, {})
    assert 'ValueError' in str(exc)
    assert 'Specification requires' in str(exc)


def test_check_fail_5():
    # Argument not in function signature
    f = check('asdf', data=[])(func_sig)
    with pytest.raises(PipelineConfigurationError) as exc:
        f({}, {'data': ['asdf']}, {})
    assert "Argument specified, 'asdf', does not" in str(exc)


def test_check_curry_1():
    # Ensure the decorator works when curried
    _func_sig = check(REQUIRE, data=[str, str])(func_sig)
    fc = toolz.curry(_func_sig,
                     **{REQUIRE: {'data': ['nir', 'red']},
                        OUTPUT: {'data': ['ndvi']}})
    fc({})


# output/requires shortcuts
def test_outputs_pass_1():
    f = requires(data=[str, str])(outputs(data=[str])(func_sig))
    f({}, {'data': ['nir', 'red']}, {'data': ['ndvi']})


# output of record must be a str
def test_output_record_str_type():
    f = outputs(record=[str])(func_sig)
    f({}, {}, {'record': ['ccdc']})

    with pytest.raises(PipelineConfigurationError) as exc:
        f = outputs(record=[str, str])(func_sig)
