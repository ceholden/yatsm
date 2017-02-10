""" Run sequence of time series algorithms or computations in a pipeline
"""
from ._pipeline import Task, Pipeline, Pipe

__all__ = [
    'Pipe',
    'Task',
    'Pipeline',
]
