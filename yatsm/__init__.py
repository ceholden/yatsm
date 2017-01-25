""" YATSM
"""
import logging

from .version import __version__

__all__ = [
    'algorithms',
    'classifiers',
    'config',
    'io',
    'mapping',
    'regression',
    'phenology'
]


# See: http://docs.python-guide.org/en/latest/writing/logging/
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(NullHandler())
