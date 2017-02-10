""" Build pipeline dependency graph from requirement configuration

Requirement configurations for a task are organized into a set of
"requires" and "outputs". These commponents are further categorized into
groups based on the type of object, either raster "data", table-like "record",
and generic Python "cache" objects.

For example, a classifier task using the NIR and Red bands, described in
YAML format:

.. code-block:: yaml

    require:
        data: [nir, red]
        cache: [classifier]
    output:
        data: [labels]

"""
from collections import defaultdict
import logging

import toposort

from yatsm.pipeline.language import (
    PIPE_CONTENTS,
    OUTPUT,
    REQUIRE,
    PIPE
)

logger = logging.getLogger(__name__)


def _format_deps(d):
    """ Return formatted list of task OUTPUT or REQUIRE

    Transform as follows:

    .. code-block:: python

        >>> d = {
            'data': ['red', 'nir', 'ndvi'],
            'record': ['ccdc'],
            'cache': ['rf']
        }  # example :ref:`REQUIRE`
        >>> _format_deps(d)
        ['data-red', 'data-nir', 'data-ndvi', 'record-ccdc', 'cache-rf']

    Args:
        d (dict): Task specification (e.g., requirements or outputs) for
            a given type (the key) and the specification (a list of str)

    Returns:
        list[str]: Formatted names of task dependencies
    """
    out = []
    for _type, names in d.items():
        out.extend(['%s-%s' % (_type, name) for name in names])
    return out


def _pipe_deps(pipe):
    """ Format data and record in a `pipe`

    Provides references to dataset bands and record information in `pipe`.

    Args:
        pipe (yatsm.pipeline.Pipe): Pipeline data

    Returns:
        dict: Dependency graph for data or results inside of `pipe`
    """
    dsk = {PIPE: set()}  # no dependencies for pipe item
    deps = _format_deps(dict((item, pipe[item].keys())
                             for item in PIPE_CONTENTS))
    for dep in deps:
        dsk[dep] = set([PIPE])
    return dsk


def _config_to_deps(config, dsk=None, overwrite=True):
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
        deps = _format_deps(spec[REQUIRE])
        dsk[task] = dsk[task].union(deps)

        # Add in items provided by task
        prov = _format_deps(spec[OUTPUT])
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
            raise KeyError('Task "{0}" has {1} unmet dependencies: {2}'
                           .format(task, len(missing), ', '.join(missing)))
    return tasks


def config_to_tasks(config, pipe, overwrite=True):
    """ Return a list of tasks from a pipeline specification

    Args:
        config (dict): Pipeline specification
        pipe (yatsm.pipeline.Pipe): Container storing `data` and `record` keys
        overwrite (bool): Allow tasks to overwrite values that have already
            been computed

    Returns:
        list: Tasks to run from the pipeline specification, given in the
            order required to fullfill all dependencies
    """
    dsk = _config_to_deps(config, dsk=_pipe_deps(pipe), overwrite=overwrite)

    tasks = [task for task in toposort.toposort_flatten(dsk)
             if task in config.keys()]

    return validate_dependencies(tasks, dsk)
