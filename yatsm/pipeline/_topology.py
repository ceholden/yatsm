""" Build pipeline dependency graph from requirements
"""
from collections import defaultdict
import logging

import six
import toposort

from .language import OUTPUT, REQUIRE, PIPE

logger = logging.getLogger(__name__)


def format_deps(d):
    """ Return formatted list of dependencies from 'requires'/'provides'

    Transform as follows:

    .. code-block:: python

        >>> d = {
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
        out.extend(['%s-%s' % (_type, name) for name in names])
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
    dsk = {PIPE: set()}
    deps = {
        'data': pipe['data'].keys(),
        'record': pipe['record'].keys()
    }
    _deps = format_deps(deps)
    for _dep in _deps:
        dsk[_dep] = set([PIPE])
    return dsk


def config_to_deps(config, dsk=None, overwrite=True):
    """ Convert a pipeline specification into list of tasks

    Args:
        config (dict): Specification of pipeline tasks
        dsk (dict): Optionally, provide a dictionary that already includes
            some dependencies. The values of this dict should be sets.
        overwrite (bool): Allow tasks to overwrite values that have already
            been computed

    Returns:
        dict: Dependency graph
    """
    dsk = defaultdict(set, dsk) if dsk else defaultdict(set)

    for task, spec in config.items():
        # Add in task requirements
        deps = format_deps(spec[REQUIRE])
        dsk[task] = dsk[task].union(deps)

        # Add in data/record provided by task
        prov = format_deps(spec[OUTPUT])
        task_needed = False
        for _prov in prov:
            if overwrite or _prov not in dsk:
                logger.debug('Adding task: {}'.format(task))
                dsk[_prov].add(task)
                task_needed = True
            else:
                logger.debug('Task already computed and not overwrite - not '
                             'adding: {}'.format(task))

        # If this task didn't provide any new data/record, cull it
        if not task_needed:
            logger.debug('Culling task {} because everything it provides is '
                         'already calculated (e.g., from cache)'.format(task))
            del dsk[task]

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
            missing_str = ', '.join(['%i) "%s"' % (i + 1, m) for i, m in
                                     enumerate(missing)])
            raise KeyError('Task "{}" has unmet dependencies: {}'
                           .format(task, missing_str))
    return tasks


def config_to_tasks(config, pipe, overwrite=True):
    """ Return a list of tasks from a pipeline specification

    Args:
        config (dict): Pipeline specification
        pipe (dict): Container storing `data` and `record` keys
        overwrite (bool): Allow tasks to overwrite values that have already
            been computed

    Returns:
        list: Tasks to run from the pipeline specification, given in the
            order required to fullfill all dependencies
    """
    _dsk = pipe_deps(pipe)
    dsk = config_to_deps(config, dsk=_dsk, overwrite=overwrite)

    tasks = [task for task in toposort.toposort_flatten(dsk)
             if task in config.keys()]

    return validate_dependencies(tasks, dsk)
