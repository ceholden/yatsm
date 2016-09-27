""" Functional wrappers around change detection algorithms
"""
import patsy

from yatsm.algorithms import CCDCesque
from .._task_validation import outputs, requires


@requires(data=[])
@outputs(record=[str])
def pixel_CCDCesque(work, require, output, **config):
    """ Run CCDCesque on a pixel
    """
    arr = work['data'][require['data']].dropna('time', how='any').to_array()

    model = CCDCesque(**config.get('init', {}))
    model.py, model.px = arr.y, arr.x

    ordinal = arr.indexes['time'].map(lambda x: x.toordinal())
    design = config.get('fit', {}).get('design', '1 + ordinal')
    X = patsy.dmatrix(design,
                      data=arr,
                      eval_env=patsy.EvalEnvironment.capture())

    model = model.fit(X, arr.values, ordinal)
    work['record'][output['record'][0]] = model.record

    return work
