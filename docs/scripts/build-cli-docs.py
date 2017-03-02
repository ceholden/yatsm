#!/usr/bin/env python
""" Build CLI help pages to RST for dynamic inclusion of help messages

This solves the problem of not being able to install YATSM on readthedocs
because of its complicated dependencies without the need to mock out
basically every import. Just run this script before pushing any new changes
to the documentation to make sure the ``yatsm [subcommand] --help`` usage
is up to date.
"""
from contextlib import contextmanager
import errno
import itertools
import os
from pathlib import Path
import subprocess
import sys

import click_plugins

from yatsm.cli.main import cli as yatsm_cli


# Add YATSM to sys.path
here = Path(os.path.dirname(__file__)).resolve()
sys.path.insert(0, os.path.join(str(here.parent)))


def make_destination(clean=True):
    def _rmtree(p):
        for _p in p.iterdir():
            _rmtree(_p) if _p.is_dir() else _p.unlink()
        p.rmdir()
    # Output directory
    help_docs_dst = here.joinpath('cli', 'usage')
    try:
        if clean:
            _rmtree(help_docs_dst)
        help_docs_dst.mkdir(parents=True)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    return help_docs_dst


@contextmanager
def redirect_stdout(stream):
    """ Redirect stdout to file to capture click's printouts

    NOTE:
        Available as contextlib.redirect_stdout in Python 3.4, but
        re-coded here for compatibility with Python 2.7.
        See https://bugs.python.org/issue15805
    """
    old_stream = sys.stdout
    sys.stdout = stream
    try:
        yield
    finally:
        sys.stdout = old_stream


def cmd_help_to_rst(cmd, dst, name):
    with open(dst, 'w') as fid:
        fid.write('$ {} --help\n'.format(name))
        with redirect_stdout(fid):
            try:
                cmd.make_context(name, ['--help'])
            except SystemExit:
                # Success
                pass


if __name__ == '__main__':
    help_docs_dst = make_destination()
    # CLICK COMMAND LINE
    for cmd in itertools.chain([yatsm_cli], yatsm_cli.commands.values()):
        if isinstance(cmd, click_plugins.core.BrokenCommand):
            continue
        name = 'yatsm {}'.format(cmd.name) if cmd.name != 'cli' else 'yatsm'
        dst = help_docs_dst.joinpath('{}.txt'.format(name.replace(' ', '_')))
        cmd_help_to_rst(cmd, dst, name)

    # SCRIPTS IN yatsm/scripts
    script_dir = here.parent.parent.joinpath('scripts')
    os.environ['PATH'] += '%s%s' % (os.pathsep, str(script_dir))
    for script in script_dir.glob('*'):  # ignores 'hidden' files
        from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA
        script_name = script_dir.joinpath(script).stem
        dst = help_docs_dst.joinpath('{}.txt'.format(script_name))
        with open(str(dst), 'w') as fid:
            fid.write('$ {} -h\n'.format(script))
            fid.flush()
            from IPython.core.debugger import Pdb; Pdb().set_trace()  # NOQA
            subprocess.Popen([script, '-h'], stdout=fid).communicate()
