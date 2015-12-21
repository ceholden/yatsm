import contextlib
import os
import tempfile

import pytest
import six
import yaml


def deep_update(orig, upd):
    """ "Deep" modify all contents in ``origin`` with values in ``upd``
    """
    for k, v in six.iteritems(upd):
        if isinstance(v, dict):
            # recursively update sub-dictionaries
            _d = deep_update(orig.get(k, {}), v)
            orig[k] = _d
        else:
            orig[k] = v
    return orig


@pytest.fixture(scope='function')
def modify_config(request):
    @contextlib.contextmanager
    def _modify_config(f, d):
        """ Overwrites yaml file ``f`` with values in ``dict`` ``d`` """
        orig = yaml.load(open(f, 'r'))
        modified = orig.copy()
        try:
            modified = deep_update(modified, d)
            tmpcfg = tempfile.mkstemp(prefix='yatsm_', suffix='.yaml')[1]
            yaml.dump(modified, open(tmpcfg, 'w'))
            yield tmpcfg
        except:
            raise
        finally:
            os.remove(tmpcfg)
    return _modify_config
