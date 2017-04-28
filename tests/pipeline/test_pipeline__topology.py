""" Tests for ``yatsm.pipeline._topology``
"""
import pytest

import yatsm.pipeline._topology as topo


@pytest.mark.parametrize(('arg', 'answer'), [
    ({'data': ['nir', 'red']}, ['data-nir', 'data-red']),
    ({'data': ['nir'], 'record': ['asdf']}, ['data-nir', 'record-asdf']),
    ({'record': ['asdf', 'fdsa']}, ['record-asdf', 'record-fdsa'])
])
def test_format_deps(arg, answer):
    assert sorted(answer) == sorted(topo._format_deps(arg))


@pytest.mark.parametrize(('arg', 'answer'), [
    ({
        'data': {'red': 5, 'nir': 3},
        'record': {'ccdc': 0}
     }, {
        'pipe': set(),
        'data-red': set(['pipe']),
        'data-nir': set(['pipe']),
        'record-ccdc': set(['pipe'])
    }),
    ({
        'data': {'nir': 1},
        'record': {}
     }, {
        'pipe': set(),
        'data-nir': set(['pipe'])
     })
])
def test_pipe_deps(arg, answer):
    deps = topo._pipe_deps(arg)
    assert sorted(deps) == sorted(answer)


def test_config_to_deps_1(tasks1):
    dsk = topo._config_to_deps(tasks1)
    for lhs, rhs in dsk.items():
        assert (lhs.startswith('data-') or lhs.startswith('record-') or
                lhs in tasks1)


def test_config_to_deps_1_dsk(tasks1):
    dsk = {
        'pipe': set([]),
        'data-red': set(['pipe']),
        'data-nir': set(['pipe'])
    }
    dsk = topo._config_to_deps(tasks1, dsk)

    assert 'pipe' in dsk
    for lhs, rsh in dsk.items():
        if lhs != 'pipe':
            assert (lhs.startswith('data-') or lhs.startswith('record-')
                    or lhs in tasks1)


def test_validate_dependencies_1(tasks1, pipe_defn):
    tasks = ['merged']
    dsk = topo._pipe_deps(pipe_defn)
    dsk = topo._config_to_deps(tasks1, dsk)

    topo.validate_dependencies(tasks, dsk)


def test_validate_dependencies_fail(tasks3, pipe_defn):
    tasks = ['ccdc']
    dsk = topo._pipe_deps(pipe_defn)
    dsk = topo._config_to_deps(tasks3, dsk)
    # This one should fail -- missing NDMI calculation
    with pytest.raises(KeyError) as exc:
        topo.validate_dependencies(tasks, dsk)
    assert 'unmet dependencies' in str(exc.value)
    assert 'ndmi' in str(exc.value)
