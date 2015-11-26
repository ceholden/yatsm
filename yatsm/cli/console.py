""" Functions for interpreter access in YATSM CLI

Function ``open_interpreter`` can be called to open up a Python or IPython
interpreter with access to a variety of objects in local memory.

TODO:
    * Add more useful functions for exploring data
"""
import code
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np

from yatsm import __version__

_funcs = locals()


def open_interpreter(model, message=None, funcs=None):
    """ Opens an (I)Python interpreter

    Args:
        model (YATSM model): Pass YATSM model to work with
        message (str, optional): Additional message to pass to user in banner
        funcs (dict of callable, optional): Functions available in (I)Python
            session

    """
    local = dict(_funcs, model=model, np=np, plt=plt)
    if funcs:
        local.update(funcs)

    banner = """\
        YATSM {yver} Interactive Interpreter (Python {pver})
        Type "help(model)" for info on YATSM model methods.
        NumPy and matplotlib.pyplot are already imported as "np" and "plt".
    """.format(
        yver=__version__,
        pver='.'.join(map(str, sys.version_info[:3])),
        funcs='\n\t'.join([k for k in local])
    )
    banner = textwrap.dedent(banner)
    if isinstance(message, str):
        banner += '\n' + message

    try:
        import IPython
        IPython.InteractiveShell.banner1 = banner
        IPython.start_ipython(argv=[], user_ns=local)
    except:
        code.interact(banner, local=local)
