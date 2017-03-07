""" Test for `yatsm.pipeline._pipeline`
"""
import pytest

from yatsm.pipeline._pipeline import Pipe

import logging
logger = logging.getLogger('yatsm')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


data = [
    {'x': 5},
    {'x': 3, 'y': 10},
    {},
    {'x': 5}
]
record = [
    None,
    None,
    {'ccdc': [None]},
    {'ccdc': [None]}
]
stash = [
    None,
    None,
    None,
    {'square': lambda x: x ** 2},
    {'square': lambda x: x ** 2, 'triple': lambda x: x ** 3}
]
inputs = pytest.mark.parametrize(
    ('data', 'record', 'stash', ),
    ((d, r, s) for (d, r, s) in zip(data, record, stash))
)

pipes = pytest.mark.parametrize(
    'pipe',
    list(Pipe(data=d, record=r, stash=s) for (d, r, s) in zip(data, record, stash))
)


@inputs
def test_Pipe_init(data, record, stash):
    pipe = Pipe(data=data, record=record, stash=stash)
    for item in pipe:
        # should init to input or dict()
        assert isinstance(item, dict)


@pipes
def test_Pipe_getitem(pipe):
    assert pipe['data'] == pipe.data
    assert pipe['record'] == pipe.record
    assert pipe['stash'] == pipe.stash

@pipes
def test_Pipe_tuple(pipe):
    assert len(pipe) == len(pipe._fields)
    for one, two in zip(pipe, (pipe.data, pipe.record, pipe.stash, )):
        assert one == two


@pipes
def test_Pipes_dictlike(pipe):
    assert list(pipe.keys()) == list(pipe.__slots__)
    assert dict(pipe.items()) == dict((k, v) for (k, v) in
                                        zip(pipe._fields, pipe))
    assert pipe.get('record') is pipe.record
    assert pipe.get('records', None) is None


@pipes
def test_Pipe_invalid(pipe):
    with pytest.raises(TypeError) as ae:
        pipe['no'] = 'no'
    assert 'does not support item assignment' in str(ae)
