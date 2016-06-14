""" Build pipeline dependency graph from requirements
"""
from collections import defaultdict

import six
import toposort


def format_deps(d):
    """ Return formatted list of dependencies from 'requires'/'provides'

    Transform as follows:

    .. code-block:: python

        >>> d = req = {
            'data': ['red', 'nir', 'ndvi'],
            'record': ['ccdc'],
        }
        >>> format_deps(d)
        ['data-red', 'data-nir', 'data-ndvi', 'record-ccdc']

    Args:
        d (dict): Task specification, requirements or provisions

    Returns:
        list: Formatted names of task dependencies
    """
    out = []
    for _type, names in six.iteritems(d):
        out.extend(['{t}-{n}'.format(t=_type, n=name) for name in names])
    return out



def pipe_deps(pipe):
    """ Format data and record in a `pipe`

    Provides references to dataset bands and record information in `pipe`.

    Args:
        pipe (dict): A "pipeline" object containing `data` and `record` as
            keys.

    Returns:
        dict: Dependency graph for data or results inside of `pipe`
    """
    dsk = {'pipe': set()}
    deps = {
        'data': pipe['data'].keys(),
        'record': pipe['record'].keys()
    }
    _deps = format_deps(deps)
    for _dep in _deps:
        dsk[_dep] = set(['pipe'])
    return dsk


def config_to_deps(config, dsk=None):
    """ Convert a pipeline specification into list of tasks

    Args:
        config (dict): Specification of pipeline tasks
        dsk (dict): Optionally, provide a dictionary that already includes
            some dependencies. The values of this dict should be sets.

    Returns:
        dict: Dependency graph
    """
    dsk = defaultdict(set, dsk) if dsk else defaultdict(set)

    for task, spec in six.iteritems(config):
        # Add in task requirements
        deps = format_deps(spec['requires'])
        dsk[task] = dsk[task].union(deps)

        # Add in data/record provided by task
        prov = format_deps(spec['provides'])
        for _prov in prov:
            dsk[_prov].add(task)

    return dsk


def validate_dependencies(tasks, dsk):
    """ Check that all required tasks are provided by `dsk`

    Args:
        tasks (Sequence[str]): Tasks to run, given in order of dependency
        dsk (dict[str, set[str]]): Dependency graph

    Returns:
        list[str]: List of input tasks

    Raises:
        KeyError: Raise if not all dependencies are met
    """
    # First validate the DAG
    for task, deps in dsk.items():
        check = [dep in dsk for dep in deps]
        if not all(check):
            missing = [dep for dep, ok in zip(deps, check) if not ok]
            raise KeyError('Task "{}" has unmet dependencies: {}'
                           .format(task, missing))
    return tasks


def config_to_tasks(config, pipe):
    """ Return a list of tasks from a pipeline specification

    Args:
        config (dict): Pipeline specification
        pipe (dict): Container storing `data` and `record` keys

    Returns:
        list: Tasks to run from the pipeline specification, given in the
            order required to fullfill all dependencies
    """
    _dsk = pipe_deps(pipe)
    dsk = config_to_deps(config, dsk=_dsk)
    tasks = toposort.toposort_flatten(dsk)

    tasks = [task for task in tasks if task in config.keys()]
    return validate_dependencies(tasks, dsk)
