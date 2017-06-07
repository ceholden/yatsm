""" Run sequence of time series algorithms or computations in a pipeline
"""
from . import language
from ._attributes import (segment_task, eager_task,
                          task_version)
from ._registry import find_tasks
from ._pipeline import Task, Pipeline, Pipe
from ._validation import outputs, requires, stash

#: dict: Pipeline tasks
PIPELINE_TASKS = find_tasks()


__all__ = [
    'Pipe',
    'Pipeline',
    'Task',
    'language',
    'eager_task',
    'segment_task',
    'task_version',
    'outputs',
    'requires',
    'stash'
]
