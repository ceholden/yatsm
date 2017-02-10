""" Functional wrappers around change detection algorithms
"""
from yatsm.algorithms import CCDCesque
from yatsm.pipeline.tasks._validation import outputs, requires, version


@version(CCDCesque.__algorithm__)
@requires(data=[])
@outputs(record=[str])
def pixel_CCDCesque(pipe, require, output, config=None):
    """ Run :class:`yatsm.algorithms.CCDCesque` on a pixel

    Users should pass to ``require`` both ``X`` and ``Y`` arguments, which
    are interpreted as:

    .. code-block:: python

        X, Y = require[0], require[1:]

    Args:
        pipe (yatsm.pipeline.Pipe): Piped data to operate on
        require (dict[str, list[str]]): Labels for the requirements of this
            calculation
        output (dict[str, list[str]]): Label for the result of this
            calculation
        config (dict): Configuration to pass to :class:`CCDCesque`. Should
            contain `init` section

    Returns:
        yatsm.pipeline.Pipe: Piped output

    """
    XY = pipe.data[require['data']].dropna('time', how='any')
    X = XY[require['data'][0]]
    Y = XY[require['data'][1:]].to_array()

    model = CCDCesque(**config.get('init', {}))
    model.py, model.px = Y.y, Y.x

    model = model.fit(X, Y.values, XY['ordinal'])
    pipe.record[output['record'][0]] = model.record

    return pipe
