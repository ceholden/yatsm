""" Run sequence of time series algorithms or computations in a pipeline
"""
from ._task_validation import eager_task, requires, outputs
from ._pipeline import Task, Pipeline

__all__ = [
    'Task',
    'Pipeline',
    'eager_task',
    'requires', 'outputs'
]
