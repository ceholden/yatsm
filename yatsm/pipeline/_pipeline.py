""" Classes representing pipeline objects
"""
from collections import OrderedDict
from functools import partial
import logging

from dask import delayed

from yatsm.errors import PipelineConfigurationError as PCError
from yatsm.pipeline._topology import config_to_tasks
from yatsm.pipeline.language import (CONFIG, TASK, REQUIRE, OUTPUT,
                                     PIPE_CONTENTS)
from yatsm.pipeline.tasks import PIPELINE_TASKS, SEGMENT_TASKS

logger = logging.getLogger(__name__)


class Pipe(object):
    """ A Bunch for pipeline communication
    """

    #: list: List of attributes stored in :class:`Pipe`
    __slots__ = PIPE_CONTENTS + ('contents', )

    def __init__(self, **kwds):
        for item in self.__slots__:
            value = kwds.get(item, dict())
            setattr(self, item, value)
        self.contents = self.__slots__[:-1]

    def pixel(self, y, x):
        """ Subset ``Pipe`` to one pixel
        """
        return Pipe(
            data=self.data.sel(y=y, x=x),
            record=self.record
        )

    # Limit possible attributes
    # TODO: reconsider? could be useful for metadata
    def __setattr__(self, key, value):
        if key not in self.__slots__:
            raise AttributeError('{0.__class__.__name__} cannot store '
                                 'attributes other than: "{1}"'
                                 .format(self, '", "'.join(PIPE_CONTENTS)))
        else:
            object.__setattr__(self, key, value)

# DICT-LIKE
    def __len__(self):
        return len(self.contents)

    def __iter__(self):
        for item in self.contents:
            yield item

    def __getitem__(self, key):
        if key in self.contents:
            return getattr(self, key)
        else:
            raise KeyError('{0.__class__.__name__} does not store '
                           'attributes other than: "{1}"'
                           .format(self, '", "'.join(PIPE_CONTENTS)))

    def items(self):
        for key in self:
            value = self[key]
            yield key, value

    def keys(self):
        for key in self:
            yield key

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __repr__(self):
        return "<{0.__class__.__name__} at {1}> storing:{2}".format(
            self,
            hex(id(self)),
            '\n'.join(self.contents)
        )


class Task(object):
    """ A :class:`Pipeline` task
    """
    def __init__(self, name, func, require, output, **config):
        self.name = name
        self.func = func
        self.funcname = self.func.__name__
        if self.func in SEGMENT_TASKS.values():
            self.create_group = True
        else:
            self.create_group = False
        self.require = require
        self.output = output
        self.config = config

    @classmethod
    def from_config(cls, name, config):
        task = config[TASK]
        try:
            func = PIPELINE_TASKS[task]
        except KeyError as ke:
            logger.exception("Unknown pipeline task '{}'".format(task), ke)
            raise
        return cls(name, func, config[REQUIRE], config[OUTPUT],
                   **config.get(CONFIG, {}))

    def curry(self):
        return partial(self.func, **self.spec)

    # TODO: maybe there is eager in time, eager in space?
    @property
    def is_eager(self):
        """ Can this task be executed on all pixels (x/y) at once?
        """
        return getattr(self.func, 'is_eager', False)

    def record_dependencies(self, tasks):
        """ Return a list of ``Task`` that this ``Task`` is dependent upon

        Args:
            tasks (list[Task]): List of ``Task`` in chain of tasks (e.g., a
                pipeline)

        Returns:
            list[Task]: Tasks with outputs that this task requires for
            computation
        """
        deps = set()
        for task in tasks:
            if task.output_record in self.require_record:
                deps.add(task)
                deps.update(task.record_dependencies(tasks))
        return list(deps)[::-1]

    def record_result(self, results):
        """ Extract this task's result from a pipe of results
        """
        return results.get(self.output_record)

    @property
    def metadata(self):
        """ dict: Attributes to store as metadata
        """
        return dict(
            config=self.spec,
            version=self.func.version
        )

# SHORTCUT GETTERS
    @property
    def spec(self):
        return {
            REQUIRE: self.require,
            OUTPUT: self.output,
            CONFIG: self.config
        }

    @property
    def require_data(self):
        return self.require.get('data', [])

    @property
    def require_record(self):
        return self.require.get('record', [])

    @property
    def output_data(self):
        return self.output.get('data', [])

    @property
    def output_record(self):
        """ str: name of 'record' output"""
        return self.output.get('record', [None])[0]

    def __repr__(self):
        return ('<{0.__class__.__name__} "{0.name}": {0.func.__name__}'
                '( {0.require} )-> {0.output} >'
                .format(self))


class Pipeline(object):
    """ A pipeline of many tasks to be executed

    Args:
        tasks (list[Task]): A list of ``Task`` to execute

    """

    def __init__(self, tasks):
        self.tasks = tasks
        self.eager_pipeline, self.pipeline = self._split_eager(self.tasks)

    @classmethod
    def from_config(cls, config, pipe, overwrite=True):
        """ Initialize a pipeline from a configuration and some data

        Args:
            config (dict): Pipeline configuration
            pipe (dict): "record" and "data" datasets
            overwrite (bool): Overwrite pre-existing results

        Returns:
            Pipeline: Pipeline of tasks
        """
        # Get sorted order based on config and input data
        task_names = config_to_tasks(config, pipe, overwrite=overwrite)
        tasks = OrderedDict(
            (name, Task.from_config(name, config[name]))
            for name in task_names
        )

        return cls(tasks)

# EXECUTION
    @staticmethod
    def delayed(tasks, pipe):
        """ Return a :class:`dask.delayed.Delayed` pipeline from tasks

        Args:
            tasks (list[Task]): List of ``Task`` in chain of tasks (e.g., a
                pipeline)
            pipe (Pipe): Pipeline data
        Returns:
            dask.delayed.Delayed: A delayed pipeline ready to be executed

        """
        pl = delayed(tasks[0].curry(), name=tasks[0].name)(
            pipe,
            dask_key_name=tasks[0].name
        )

        for task in tasks[1:]:
            pl = delayed(task.curry(), name=task.name)(
                pl,
                dask_key_name=task.name
            )

        return pl

    # TODO: test without using dask.delayed, just function wrapping
    #       ... curious about the dask overhead
    def run_eager(self, pipe, **compute_kwds):
        """ Run eager steps

        Args:
            pipe (Pipe): Pipeline data

        """
        pipeline = self.delayed(self.eager_pipeline, pipe)
        return pipeline.compute(**compute_kwds)

    def run(self, pipe, check_eager=True, **compute_kwds):
        """ Run all steps

        Args:
            pipe (Pipe): Pipeline data

        """
        # Check if pipe contains "eager" pipeline outputs
        if check_eager and not self._check_eager(self.eager_pipeline, pipe):
            logger.warning('Triggering eager compute')
            pipe = self.run_eager(pipe)
        pipeline = self.delayed(self.pipeline, pipe)
        return pipeline.compute(**compute_kwds)

    def visualize(self, pipe, filename=None, rankdir='LR', **visualize_kwds):
        """ Show the Dask task graph as a Graphviz plot

        Args:
            pipe (Pipe): Pipeline data
            filename (str):The name (without an extension) of the file to write
                to disk. By default, no file is written (``filename=None``)
            rankdir (str): Layout direction of graph
            **visualize_kwds (dict): Options passed to :func:`dask.visualize`
                (see :func:`dask.dot.to_graphviz`)

        Returns:
            IPython.core.display.Image: Pipeline visualized using
            Graphviz, or ``None`` if :mod:`IPython` is not installed.
        """
        pl = self.delayed(self.eager_pipeline + self.pipeline, pipe)
        return pl.visualize(rankdir=rankdir, **visualize_kwds)

# TASK STORAGE
    def task_table(self, task):
        """ Define what HDF5 group/table this task belongs in

        Args:
            tasks (list[Task]): List of ``Task`` in chain of tasks (e.g., a
                pipeline)

        Returns:
            tuple (str, str): Return the group root (e.g., '/ccdc') and the
            table name (e.g., 'ccdc') if the task outputs a table, otherwise
            (None, None)
        """
        if not task.output_record:
            return (None, None)

        group = [_task.name for _task in
                 task.record_dependencies(self.tasks.values())
                 if task.create_group]

        where = '/'
        if group:
            where += '/'.join(group)
        elif task.create_group:
            where += '%s' % task.name
        else:
            raise PCError("Task '{}' has no root segment and does not create "
                          "a segment".format(task.name))

        return (where, task.name)

    @property
    def task_tables(self):
        """ dict: :ref:`Task` -> result file table location (str)
        """
        return dict((task, self.task_table(task)) for task in
                    self.tasks.values() if task.output_record)

# HELPERS
    @staticmethod
    def _check_eager(tasks, pipe):
        """ Check if it looks like eager task results have been computed
        """
        for eager_task in tasks:
            data, rec = eager_task.output_data, eager_task.output_record
            has_data = [output in pipe['data'] for output in data]
            has_rec = rec in pipe['record']
            if not all(has_data) and not has_rec:
                missing = [item for item, has in zip(data, has_data)
                           if not has]
                if not has_rec:
                    missing.append(rec)
                logger.warning('Eager task {t} has missing output: {m}'
                               .format(t=eager_task.funcname, m=missing))
                return False
        return True

    @staticmethod
    def _split_eager(tasks):
        halt_eager = False
        eager_pipeline, pipeline = [], []
        for task in tasks.values():
            if task.is_eager and not halt_eager:
                eager_pipeline.append(task)
            else:
                if task.is_eager:
                    logger.debug('Not able to compute eager function "{}" on '
                                 'all pixels at once  because it came after '
                                 'non-eager tasks.'.format(task.funcname))
                    halt_eager = True
                pipeline.append(task)

        return eager_pipeline, pipeline
