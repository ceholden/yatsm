""" YATSM
"""
import logging

from yatsm.version import __version__


# See: http://docs.python-guide.org/en/latest/writing/logging/
import logging
try:  # Python 2.7+
    from logging import NullHandler as _NullHandler
except ImportError:
    class _NullHandler(logging.Handler):
        def emit(self, record):
            pass
logging.getLogger(__name__).addHandler(_NullHandler())
